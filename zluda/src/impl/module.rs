use std::{
    collections::hash_map, collections::HashMap, ffi::c_void, ffi::CStr, ffi::CString, mem,
    os::raw::c_char, ptr, slice,
};

use super::{
    device,
    function::Function,
    function::{FunctionData, LegacyArguments},
    CUresult, GlobalState, HasLivenessCookie, LiveCheck,
};
use ptx;

pub type Module = LiveCheck<ModuleData>;

impl HasLivenessCookie for ModuleData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0xf1313bd46505f98a;

    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0xbdbe3f15;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        Ok(())
    }
}

pub struct ModuleData {
    pub spirv: SpirvModule,
    /// The argument sizes of the kernels that have been translated so far.
    pub kernel_info: HashMap<String, ptx::KernelInfo>,
    // This should be a Vec<>, but I'm feeling lazy
    pub device_binaries: HashMap<device::Index, CompiledModules>,
}

pub struct SpirvModule {
    /// The PTX text of every record of the source image. A fatbin carries one
    /// record per translation unit, and a record can define hundreds of
    /// kernels: translating all of them costs minutes per record and a run
    /// uses a handful, so the text is kept and one kernel at a time is
    /// translated (see `ptx_split`).
    pub ptx_sources: Vec<String>,
    /// The record of the image that defines a kernel.
    pub kernel_units: HashMap<String, usize>,
}

/// One kernel, translated and ready to be compiled.
pub struct TranslatedKernel {
    pub binary: Vec<u32>,
    /// The kernel calls into the PTX implementation module, which has to be
    /// linked in; the others are built on their own.
    pub need_ptx_impl: bool,
    pub ptx_impl: Option<&'static [u8]>,
    pub info: ptx::KernelInfo,
    pub build_options: CString,
}

pub struct CompiledModule {
    pub base: l0::Module<'static>,
    pub kernels: HashMap<CString, Box<Function>>,
}

/// The modules of a fatbin, one per translation unit. They are built
/// separately: linking all of them into one needs a lot of memory, and the
/// kernels of a translation unit do not call into the other ones.
/// The modules of a fatbin, one per translation unit. They are built when
/// one of their kernels is used: building all of them takes hours on a
/// program with a hundred units, and most of them are never used.
pub struct CompiledModules {
    pub kernels: HashMap<String, CompiledModule>,
}

impl<L, T, E> From<ptx::ParseError<L, T, E>> for CUresult {
    fn from(_: ptx::ParseError<L, T, E>) -> Self {
        CUresult::CUDA_ERROR_INVALID_PTX
    }
}

impl From<ptx::TranslateError> for CUresult {
    fn from(_: ptx::TranslateError) -> Self {
        CUresult::CUDA_ERROR_INVALID_PTX
    }
}


impl SpirvModule {
    pub fn new_raw<'a>(text: *const c_char) -> Result<Self, CUresult> {
        let u8_text = unsafe { CStr::from_ptr(text) };
        let ptx_text = u8_text
            .to_str()
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_PTX)?;
        Self::new(ptx_text)
    }

    pub fn new<'a>(ptx_text: &str) -> Result<Self, CUresult> {
        Self::new_many(&[ptx_text])
    }

    /// Translate one or more PTX modules into a single SPIR-V module.
    ///
    /// A fatbin usually holds one PTX record per translation unit. The CUDA
    /// runtime JITs all of them and links the result, so the kernels of every
    /// record end up in one module. We do the same: each record is
    /// translated and the SPIR-V modules are linked together at compile time.
    pub fn new_many<'a>(ptx_modules: &[&str]) -> Result<Self, CUresult> {
        let mut ptx_sources = Vec::with_capacity(ptx_modules.len());
        let mut kernel_units = HashMap::new();
        for (index, ptx_text) in ptx_modules.iter().enumerate() {
            // Only the entry point names are needed here: the kernels are
            // translated one at a time, when one of them is asked for.
            for name in super::ptx_split::entry_names(ptx_text) {
                kernel_units.entry(name).or_insert(index);
            }
            ptx_sources.push(ptx_text.to_string());
        }
        if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
            eprintln!(
                "new_many: {} records, {} kernels",
                ptx_sources.len(),
                kernel_units.len()
            );
        }
        Ok(SpirvModule {
            ptx_sources,
            kernel_units,
        })
    }

    /// Translate one kernel of the image: its entry point and the functions it
    /// reaches, and nothing else.
    pub fn translate_kernel(
        &self,
        unit: usize,
        kernel: &str,
    ) -> Result<TranslatedKernel, CUresult> {
        let text = self
            .ptx_sources
            .get(unit)
            .ok_or(CUresult::CUDA_ERROR_INVALID_PTX)?;
        // ZLUDA_PTX_NO_SPLIT translates the whole record, the way it was done
        // before the split existed. It is the reference for bisecting a
        // difference between the two, and it is slow.
        let split = if std::env::var_os("ZLUDA_PTX_NO_SPLIT").is_some() {
            text.to_string()
        } else {
            match super::ptx_split::split_for_kernel(text, kernel) {
                Some(split) => split,
                None => {
                    // Either the record does not have that kernel, or the
                    // split would have left a call without its definition.
                    // Translating the whole record is correct in both cases
                    // (and slow), so it is the fallback rather than a guess.
                    if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                        eprintln!(
                            "kernel {:?} is translated as a whole record (it is not in the \
                             split, or the split would be incomplete)",
                            kernel
                        );
                    }
                    text.to_string()
                }
            }
        };
        let trace = std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some();
        let mut errors = Vec::new();
        let ast = ptx::ModuleParser::new()
            .parse(&mut errors, split.as_str())
            .map_err(|e| {
                if trace {
                    eprintln!("kernel {:?} failed to parse: {:?}", kernel, e);
                }
                CUresult::from(e)
            })?;
        // The parser recovers from what it cannot read by dropping it, which
        // would silently compute the wrong thing, so a kernel with parse
        // errors is not loaded at all.
        if !errors.is_empty() {
            if trace || std::env::var_os("ZLUDA_PTX_ALLOW_UNPARSED").is_some() {
                eprintln!(
                    "kernel {:?} has {} unparsed constructs, first: {:?}",
                    kernel,
                    errors.len(),
                    errors.first()
                );
            }
            if std::env::var_os("ZLUDA_PTX_ALLOW_UNPARSED").is_none() {
                return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
            }
        }
        let spirv_module = ptx::to_spirv_module(ast).map_err(|e| {
            if trace {
                eprintln!("kernel {:?} failed to translate: {:?}", kernel, e);
            }
            CUresult::from(e)
        })?;
        let binary = spirv_module.assemble();
        let info = spirv_module
            .kernel_info
            .get(kernel)
            .cloned()
            .ok_or(CUresult::CUDA_ERROR_NOT_FOUND)?;
        Ok(TranslatedKernel {
            binary,
            need_ptx_impl: spirv_module.should_link_ptx_impl.is_some(),
            ptx_impl: spirv_module.should_link_ptx_impl,
            info,
            build_options: spirv_module.build_options,
        })
    }

    /// Compile a translated kernel into a Level Zero module.
    pub fn build_kernel(
        ctx: &'static l0::Context,
        dev: l0::Device,
        translated: &TranslatedKernel,
    ) -> Result<l0::Module<'static>, CUresult> {
        let bytes = unsafe {
            slice::from_raw_parts(
                translated.binary.as_ptr() as *const u8,
                translated.binary.len() * mem::size_of::<u32>(),
            )
        };
        let built: (Result<l0::Module<'static>, l0::sys::ze_result_t>, Option<l0::BuildLog>) =
            if translated.need_ptx_impl {
            let ptx_impl = translated
                .ptx_impl
                .ok_or(CUresult::CUDA_ERROR_NOT_SUPPORTED)?;
            l0::Module::build_link_spirv(
                ctx,
                dev,
                &[ptx_impl, bytes],
                Some(translated.build_options.as_c_str()),
            )
        } else {
            let (module, log) = l0::Module::build_spirv_logged(
                ctx,
                dev,
                bytes,
                Some(translated.build_options.as_c_str()),
            );
            (module, Some(log))
        };
        let (module, log) = built;
        if module.is_err() && std::env::var_os("ZLUDA_TRACE_BUILD").is_some() {
            eprintln!(
                "module build failed: {:?}\nbuild log: {}",
                module.as_ref().err(),
                log.as_ref()
                    .map(|log| log.to_cstring().unwrap_or_default().to_string_lossy().to_string())
                    .unwrap_or_default()
            );
        }
        Ok(module?)
    }


}


pub fn get_function(
    hfunc: *mut *mut Function,
    hmod: *mut Module,
    name: *const c_char,
) -> Result<(), CUresult> {
    if hfunc == ptr::null_mut() || hmod == ptr::null_mut() || name == ptr::null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let name = unsafe { CStr::from_ptr(name) }.to_owned();
    let trace = std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some();
    let function: *mut Function = GlobalState::lock_current_context(|ctx| {
        if trace {
            eprintln!("cuModuleGetFunction({:?})", name);
        }
        let module = unsafe { &mut *hmod }.as_result_mut()?;
        let device = unsafe { &mut *ctx.device };
        let kernel_name = unsafe { std::str::from_utf8_unchecked(name.as_bytes()) }.to_string();
        let compiled_modules = module
            .device_binaries
            .entry(device.index)
            .or_insert_with(|| CompiledModules {
                kernels: HashMap::new(),
            });
        // The kernel is translated and compiled the first time it is asked
        // for: a record can define hundreds of kernels and a run uses a few.
        if !compiled_modules.kernels.contains_key(&kernel_name) {
            let unit = match module.spirv.kernel_units.get(&kernel_name).copied() {
                Some(unit) => unit,
                None => {
                    if trace {
                        eprintln!(
                            "kernel {:?} is not in the image ({} kernels)",
                            kernel_name,
                            module.spirv.kernel_units.len()
                        );
                    }
                    return Err(CUresult::CUDA_ERROR_NOT_FOUND);
                }
            };
            let translated = match module.spirv.translate_kernel(unit, &kernel_name) {
                Ok(translated) => translated,
                Err(e) => {
                    if trace {
                        eprintln!(
                            "translate of {:?} (record {}) failed: {:?}",
                            kernel_name, unit, e
                        );
                    }
                    return Err(e);
                }
            };
            let base = match SpirvModule::build_kernel(&device.l0_context, device.base, &translated) {
                Ok(base) => base,
                Err(e) => {
                    if trace {
                        eprintln!("build of {:?} failed: {:?}", kernel_name, e);
                    }
                    return Err(e);
                }
            };
            compiled_modules.kernels.insert(
                kernel_name.clone(),
                CompiledModule {
                    base,
                    kernels: HashMap::new(),
                },
            );
            module.kernel_info.insert(kernel_name.clone(), translated.info);
        }
        let kernel_info = module
            .kernel_info
            .get(&kernel_name)
            .cloned()
            .ok_or(CUresult::CUDA_ERROR_NOT_FOUND)?;
        let compiled = module
            .device_binaries
            .get_mut(&device.index)
            .and_then(|modules| modules.kernels.get_mut(&kernel_name))
            .ok_or(CUresult::CUDA_ERROR_NOT_FOUND)?;
        // A function handle stays valid for the life of the module: building a
        // second one under the same name would free the box the application
        // already holds, and the heap reusing that address turns the stale
        // handle into a different kernel.
        if let Some(existing) = compiled.kernels.get(&name) {
            return Ok::<_, CUresult>(existing.as_ref() as *const Function as *mut Function);
        }
        // The kernel is created in the unit that has it.
        let l0_kernel = match l0::Kernel::new_resident(&compiled.base, name.as_c_str()) {
            Ok(kernel) => kernel,
            Err(e) => {
                if trace {
                    eprintln!("kernel {:?} not found in its unit: {:?}", name, e);
                }
                return Err(CUresult::from(e));
            }
        };
        l0_kernel.set_indirect_access(
            l0::sys::ze_kernel_indirect_access_flags_t::ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE
            | l0::sys::ze_kernel_indirect_access_flags_t::ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST
            | l0::sys::ze_kernel_indirect_access_flags_t::ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED
        )?;
        let mut function = Box::new(Function::new(FunctionData {
            base: l0_kernel,
            arg_size: kernel_info.arguments_sizes.clone(),
            use_shared_mem: kernel_info.uses_shared_mem,
            properties: None,
            legacy_args: LegacyArguments::new(),
            name: name.clone(),
        }));
        let ptr = function.as_mut() as *mut _;
        compiled.kernels.insert(name, function);
        Ok::<_, CUresult>(ptr)
    })??;
    unsafe { *hfunc = function };
    Ok(())
}

pub(crate) fn load_data(pmod: *mut *mut Module, image: *const c_void) -> Result<(), CUresult> {
    // The image may be plain PTX text, a fatbinary container (bare header
    // magic 0xBA55ED50) or a fatbinc wrapper (0x466243B1). extract_ptx sniffs
    // all three and passes plain PTX through unchanged, so runtime-API apps
    // whose libcudart lazily calls cuModuleLoadData(Ex) with an embedded
    // fatbin get their PTX compiled instead of failing INVALID_PTX.
    if pmod == ptr::null_mut() || image == ptr::null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let ptx_modules = unsafe { super::fatbin::extract_ptx_all(image) }?;
    let ptx_texts: Vec<&str> = ptx_modules
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            std::str::from_utf8(bytes).map_err(|_| {
                if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                    eprintln!(
                        "PTX module {} is not valid UTF-8 ({} bytes): {:02x?}",
                        index,
                        bytes.len(),
                        &bytes[..bytes.len().min(32)]
                    );
                }
                CUresult::CUDA_ERROR_INVALID_PTX
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let spirv_data = SpirvModule::new_many(&ptx_texts)?;
    load_data_impl(pmod, spirv_data)
}

pub(crate) fn load_fatbin(pmod: *mut *mut Module, image: *const c_void) -> Result<(), CUresult> {
    if pmod == ptr::null_mut() || image == ptr::null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let ptx_modules = unsafe { super::fatbin::extract_ptx_all(image) }?;
    if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
        let names: Vec<String> = ptx_modules
            .iter()
            .map(|m| {
                let text = String::from_utf8_lossy(m);
                text.lines()
                    .find(|l| l.trim_start().starts_with(".visible .entry") || l.trim_start().starts_with(".entry"))
                    .map(|l| l.split_whitespace().nth(2).unwrap_or("?").to_string())
                    .unwrap_or_else(|| "?".to_string())
            })
            .collect();
        eprintln!("load_fatbin: {} PTX modules: {:?}", ptx_modules.len(), names);
    }
        let mut ptx_modules = ptx_modules;
    rename_duplicate_kernels(&mut ptx_modules);
    let ptx_texts: Vec<&str> = ptx_modules
        .iter()
        .enumerate()     .map(|(index, bytes)| {
            std::str::from_utf8(bytes).map_err(|_| {
                if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                    eprintln!(
                        "PTX module {} is not valid UTF-8 ({} bytes): {:02x?}",
                        index,
                        bytes.len(),
                        &bytes[..bytes.len().min(32)]
                    );
                }
                CUresult::CUDA_ERROR_INVALID_PTX
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let spirv_data = match SpirvModule::new_many(&ptx_texts) {
        Ok(m) => m,
        Err(e) => {
            if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                eprintln!("load_fatbin: translate failed: {:?}", e);
            }
            return Err(e);
        }
    };
    // The parser recovers from statements it cannot read by dropping them; a
    // kernel whose body does not parse would then be missing from the module
    // and only fail at launch with "named symbol not found". Check that every
    // entry point of the PTX made it into the translated module and fail the
    // load otherwise: CUDA fails the JIT of a module it cannot compile too.
    let mut missing: Vec<String> = Vec::new();
    for text in &ptx_texts {
        let mut rest: &str = text;
        while let Some(pos) = rest.find(".entry ") {
            let after = &rest[pos + ".entry ".len()..];
            let name_end = after
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$'))
                .unwrap_or(after.len());
            let name = &after[..name_end];
            if !name.is_empty() && !spirv_data.kernel_units.contains_key(name) {
                missing.push(name.to_string());
            }
            rest = &after[name_end..];
        }
    }
    if !missing.is_empty() {
        if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
            eprintln!("load: kernels that did not translate: {}", missing.len());
        }
        return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
    }
    match load_data_impl(pmod, spirv_data) {
        Ok(()) => Ok(()),
        Err(e) => {
            if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
                eprintln!("load_fatbin: module build failed: {:?}", e);
            }
            Err(e)
        }
    }
}

/// The same kernel can be defined in several translation units: a template
/// in a header (CUB's empty kernel, for example) is instantiated in every
/// unit that includes it. CUDA's device linker keeps one copy of such a
/// kernel; the OpenCL runtime's linker rejects the duplicate names, so the
/// copies after the first are renamed.
fn rename_duplicate_kernels(ptx_modules: &mut [Vec<u8>]) {
    let mut seen: Vec<String> = Vec::new();
    let mut count = 0usize;
    for module in ptx_modules.iter_mut() {
        let text = match std::str::from_utf8(module) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let mut renamed = String::with_capacity(text.len());
        let mut rest = text;
        let mut renamed_count = 0usize;
        while let Some(entry) = rest.find(".entry ") {
            let (before, after) = rest.split_at(entry + ".entry ".len());
            let name_end = after
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$'))
                .unwrap_or(after.len());
            let name = &after[..name_end];
            renamed.push_str(before);
            if seen.iter().any(|s| s == name) {
                count += 1;
                renamed_count += 1;
                renamed.push_str(&format!("{}_zluDA_dup{}", name, count));
            } else {
                seen.push(name.to_string());
                renamed.push_str(name);
            }
            rest = &after[name_end..];
        }
        renamed.push_str(rest);
        if renamed_count > 0 {
            *module = renamed.into_bytes();
        }
    }
}

pub fn load_data_impl(pmod: *mut *mut Module, spirv_data: SpirvModule) -> Result<(), CUresult> {
    let module = GlobalState::lock_current_context(|ctx| {
        let device = unsafe { &mut *ctx.device };
        // Kernels are translated and built when they are first used; see
        // get_function.
        let mut device_binaries = HashMap::new();
        device_binaries.insert(
            device.index,
            CompiledModules {
                kernels: HashMap::new(),
            },
        );
        let module_data = ModuleData {
            spirv: spirv_data,
            kernel_info: HashMap::new(),
            device_binaries,
        };
        Ok::<_, CUresult>(module_data)
    })??;
    let module_ptr = Box::into_raw(Box::new(Module::new(module)));
    unsafe { *pmod = module_ptr };
    Ok(())
}

pub(crate) fn unload(module: *mut Module) -> Result<(), CUresult> {
    if module == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock(|_| Module::destroy_impl(module))?
}

pub(crate) fn load(pmod: *mut *mut Module, fname: *const i8) -> Result<(), CUresult> {
    if pmod == ptr::null_mut() || fname == ptr::null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let path = unsafe { CStr::from_ptr(fname) };
    let path_utf8 = path
        .to_str()
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
    let file = std::fs::read(path_utf8).map_err(|_| CUresult::CUDA_ERROR_FILE_NOT_FOUND)?;
    let module_text = std::str::from_utf8(&file).map_err(|_| CUresult::CUDA_ERROR_INVALID_PTX)?;
    let spirv_data = SpirvModule::new(module_text)?;
    load_data_impl(pmod, spirv_data)
}

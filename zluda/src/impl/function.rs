use super::{stream::Stream, CUresult, GlobalState, HasLivenessCookie, LiveCheck};
use crate::cuda::CUfunction_attribute;
use ::std::os::raw::{c_uint, c_void};
use std::{hint, ptr};

const CU_LAUNCH_PARAM_END: *mut c_void = 0 as *mut _;
const CU_LAUNCH_PARAM_BUFFER_POINTER: *mut c_void = 1 as *mut _;
const CU_LAUNCH_PARAM_BUFFER_SIZE: *mut c_void = 2 as *mut _;

pub type Function = LiveCheck<FunctionData>;

impl HasLivenessCookie for FunctionData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0x5e2ab14d5840678e;

    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0x33e6a1e6;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        Ok(())
    }
}

pub struct FunctionData {
    pub base: l0::Kernel<'static>,
    /// The sizes of the arguments as the application passes them. A kernel
    /// that reads them from a buffer still receives them one by one.
    pub arg_size: Vec<usize>,
    /// Set when the argument list is too large for the compiler: the layout of
    /// the buffer the kernel reads its arguments from instead.
    pub param_buffer: Option<super::param_buffer::ParamBuffer>,
    pub use_shared_mem: bool,
    /// Bytes the kernel's own addressing assumes before the dynamic shared
    /// window: a lifted kernel declares its whole shared memory as one symbol,
    /// and the static part of it becomes the prefix.
    pub shared_mem_prefix: usize,
    pub properties: Option<Box<l0::sys::ze_kernel_properties_t>>,
    pub legacy_args: LegacyArguments,
    /// The name the kernel was looked up with, for the launch trace.
    pub name: std::ffi::CString,
}

pub struct LegacyArguments {
    block_shape: Option<(i32, i32, i32)>,
}

impl LegacyArguments {
    pub fn new() -> Self {
        LegacyArguments { block_shape: None }
    }

    #[allow(dead_code)]
    pub fn is_initialized(&self) -> bool {
        self.block_shape.is_some()
    }

    pub fn reset(&mut self) {
        self.block_shape = None;
    }
}

impl FunctionData {
    fn get_properties(&mut self) -> Result<&l0::sys::ze_kernel_properties_t, l0::sys::ze_result_t> {
        if let None = self.properties {
            self.properties = Some(self.base.get_properties()?)
        }
        match self.properties {
            Some(ref props) => Ok(props.as_ref()),
            None => unsafe { hint::unreachable_unchecked() },
        }
    }
}

pub fn launch_kernel(
    f: *mut Function,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    hstream: *mut Stream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<(), CUresult> {
    let trace = std::env::var_os("ZLUDA_TRACE_FUNC").is_some();
    if f == ptr::null_mut()
        || (kernel_params == ptr::null_mut() && extra == ptr::null_mut())
        || (kernel_params != ptr::null_mut() && extra != ptr::null_mut())
    {
        if trace {
            eprintln!(
                "launch_kernel: bad arguments f={:?} kernel_params={:?} extra={:?}",
                f, kernel_params, extra
            );
        }
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    if trace {
        if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
            eprintln!(
                "launch_kernel: {} grid=({},{},{}) block=({},{},{}) shared={}",
                func.name.to_string_lossy(),
                grid_dim_x,
                grid_dim_y,
                grid_dim_z,
                block_dim_x,
                block_dim_y,
                block_dim_z,
                shared_mem_bytes
            );
        }
    }
    // A kernel whose arguments do not fit in the parameter list reads them from
    // a buffer. Gathering the bytes is plain host work; filling a buffer here
    // rather than inside the lock is deliberate, because a device allocation
    // takes the global state lock and the launch path holds the stream.
    let mut param_block = None;
    let mut param_bytes = None;
    let function_key = f as usize;
    // ZLUDA_PARAM_BLOCKING=1 restores the old path: allocate a buffer per launch
    // and fill it with a copy that drains the context. Slow (2.5 ms per launch on
    // torch's cat) but kept as the reference the staged path is measured against.
    let blocking = std::env::var_os("ZLUDA_PARAM_BLOCKING").is_some();
    if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
        if let Some(layout) = func.param_buffer.clone() {
            if blocking {
                param_block = Some(super::param_buffer::publish(&layout, kernel_params, extra)?);
            } else {
                super::param_buffer::reserve(&layout, function_key)?;
                param_bytes = Some((
                    layout.clone(),
                    super::param_buffer::collect(&layout, kernel_params, extra)?,
                ));
            }
        }
    }
    // A launch with no parameter buffer has nothing to wait for.
    let keep_pending = if param_bytes.is_some() {
        super::param_buffer::SLOTS
    } else {
        usize::MAX
    };
    let timing = std::env::var_os("ZLUDA_TRACE_LAUNCH_MS").is_some();
    let t_enter = std::time::Instant::now();
    GlobalState::lock_enqueue_staged(hstream, keep_pending, |cmd_list, signal, wait| {
        let t_locked = t_enter.elapsed();
        let func: &mut FunctionData = match unsafe { &mut *f }.as_result_mut() {
            Ok(func) => func,
            Err(e) => {
                if trace {
                    eprintln!("launch_kernel: handle {:?} is not a live function: {:?}", f, e);
                }
                return Err(e);
            }
        };
        // ZLUDA_DUMP_PARAMS=<substring> prints the bytes of every argument of a
        // matching launch. A kernel that only misbehaves for some inputs cannot
        // be diagnosed from the source alone, and the parameter block is the
        // only place the difference shows.
        let dump_this = std::env::var("ZLUDA_DUMP_PARAMS")
            .map(|want| func.name.to_string_lossy().contains(want.as_str()))
            .unwrap_or(false);
        let traced_stage = std::env::var_os("ZLUDA_TRACE_STAGE").is_some();
        if let Some((layout, bytes)) = param_bytes.take() {
            // The slot pool rotates, so this launch gets a buffer that no
            // launch still in flight can be reading. Its copy is appended to
            // this list, ahead of the launch, which is all the ordering a
            // single queue needs.
            let t_stage_start = std::time::Instant::now();
            match super::param_buffer::stage(&layout, &bytes, function_key, cmd_list) {
                Ok(block) => {
                    if traced_stage {
                        eprintln!("launch: staged block {:p} for {}", block, func.name.to_string_lossy());
                    }
                    param_block = Some(block);
                    if timing {
                        eprintln!(
                            "launch ms: lock {:.3} stage {:.3}",
                            t_locked.as_secs_f64() * 1e3,
                            t_stage_start.elapsed().as_secs_f64() * 1e3
                        );
                    }
                }
                Err(error) => {
                    eprintln!(
                        "launch: staging failed for {} -> {:?}",
                        func.name.to_string_lossy(),
                        error
                    );
                    return Err(error);
                }
            }
        }
        if let Some(block) = param_block {
            // The whole argument list is one eight byte value: the address of
            // the buffer it was copied into.
            unsafe { func.base.set_arg_raw(0, std::mem::size_of::<*mut c_void>(), &block as *const _ as *const c_void)? };
        } else if kernel_params != ptr::null_mut() {
            for (i, arg_size) in func.arg_size.iter().enumerate() {
                let value = unsafe { *kernel_params.add(i) };
                if dump_this {
                    if value.is_null() {
                        eprintln!("param[{}] size={} null", i, arg_size);
                    } else {
                        let bytes =
                            unsafe { std::slice::from_raw_parts(value as *const u8, *arg_size) };
                        eprintln!("param[{}] size={} {:02x?}", i, arg_size, bytes);
                    }
                }
                if let Err(e) = unsafe { func.base.set_arg_raw(i as u32, *arg_size, value) } {
                    if trace {
                        eprintln!(
                            "launch_kernel: set_arg_raw({}, {}) failed: {:?}",
                            i, arg_size, e
                        );
                    }
                    return Err(e.into());
                }
            }
        } else {
            let mut offset = 0;
            let mut buffer_ptr = None;
            let mut buffer_size = None;
            loop {
                match unsafe { *extra.add(offset) } {
                    CU_LAUNCH_PARAM_END => break,
                    CU_LAUNCH_PARAM_BUFFER_POINTER => {
                        buffer_ptr = Some(unsafe { *extra.add(offset + 1) as *mut u8 });
                    }
                    CU_LAUNCH_PARAM_BUFFER_SIZE => {
                        buffer_size = Some(unsafe { *(*extra.add(offset + 1) as *mut usize) });
                    }
                    _ => {
                        if trace {
                            eprintln!("launch_kernel: unknown launch parameter");
                        }
                        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
                    }
                }
                offset += 2;
            }
            match (buffer_size, buffer_ptr) {
                (Some(buffer_size), Some(buffer_ptr)) => {
                    let sum_of_kernel_argument_sizes =
                        func.arg_size.iter().fold(0, |offset, size_of_arg| {
                            size_of_arg + round_up_to_multiple(offset, *size_of_arg)
                        });
                    if buffer_size < sum_of_kernel_argument_sizes {
                        if trace {
                            eprintln!(
                                "launch_kernel: buffer too small {} < {}",
                                buffer_size, sum_of_kernel_argument_sizes
                            );
                        }
                        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
                    }
                    let mut offset = 0;
                    for (i, arg_size) in func.arg_size.iter().enumerate() {
                        let buffer_offset = round_up_to_multiple(offset, *arg_size);
                        unsafe {
                            func.base.set_arg_raw(
                                i as u32,
                                *arg_size,
                                buffer_ptr.add(buffer_offset) as *const _,
                            )?
                        };
                        offset = buffer_offset + *arg_size;
                    }
                }
                _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
            }
        }
        if func.use_shared_mem {
            // The shared-memory size follows the kernel's own parameters, and a
            // kernel that reads its arguments from a buffer has exactly one.
            let own_parameters = if func.param_buffer.is_some() {
                1
            } else {
                func.arg_size.len()
            };
            // The application asks for its own dynamic shared memory; the
            // prefix the kernel assumes sits in front of it.
            let dynamic_shared = shared_mem_bytes as usize + func.shared_mem_prefix;
            unsafe {
                func.base.set_arg_raw(own_parameters as u32, dynamic_shared, ptr::null())?
            };
        }
        if let Err(e) = func.base.set_group_size(block_dim_x, block_dim_y, block_dim_z) {
            if trace {
                eprintln!(
                    "launch_kernel: set_group_size({}, {}, {}) failed: {:?}",
                    block_dim_x, block_dim_y, block_dim_z, e
                );
            }
            return Err(e.into());
        }
        func.legacy_args.reset();
        unsafe {
            if let Err(e) = cmd_list.append_launch_kernel(
                &mut func.base,
                &[grid_dim_x, grid_dim_y, grid_dim_z],
                Some(signal),
                wait,
            ) {
                if trace {
                    eprintln!(
                        "launch_kernel: append_launch_kernel grid ({}, {}, {}) failed: {:?}",
                        grid_dim_x, grid_dim_y, grid_dim_z, e
                    );
                }
                return Err(e.into());
            }
        }
        Ok::<_, CUresult>(())
    })?;
    // ZLUDA_SYNC_LAUNCH makes every launch block until it retires, a
    // diagnostic: a difference with it on separates a missing ordering from a
    // wrong value.
    if std::env::var_os("ZLUDA_TRACE_STREAM").is_some() {
        if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
            super::stream::remember_launched_kernel(&func.name.to_string_lossy());
        }
    }
    if std::env::var_os("ZLUDA_SYNC_LAUNCH").is_some() {
        crate::r#impl::stream::synchronize(hstream)?;
    }
    Ok(())
}

fn round_up_to_multiple(x: usize, multiple: usize) -> usize {
    ((x + multiple - 1) / multiple) * multiple
}

pub(crate) fn get_attribute(
    pi: *mut i32,
    attrib: CUfunction_attribute,
    func: *mut Function,
) -> Result<(), CUresult> {
    if pi == ptr::null_mut() || func == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let trace = std::env::var_os("ZLUDA_TRACE_FUNC").is_some();
    let value = GlobalState::lock_function(func, |func| {
        let props = match func.get_properties() {
            Ok(p) => p,
            Err(e) => {
                if trace {
                    eprintln!("cuFuncGetAttribute: kernel properties failed: {:?}", e);
                }
                return Err(CUresult::from(e));
            }
        };
        Ok::<_, CUresult>(match attrib {
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK => {
                props.maxSubgroupSize * props.maxNumSubgroups
            }
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES => props.localMemSize,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES => 0,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES => props.privateMemSize,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS => 0,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION => 80,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => 80,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA => 0,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES => {
                props.localMemSize
            }
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => {
                props.localMemSize
            }
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX => 0,
            // Attributes that are not modelled (the cluster scheduling ones,
            // for instance) read as zero, the way CUDA reports an attribute
            // that does not apply. Returning "not supported" breaks
            // cudaFuncGetAttributes, which fills a whole struct.
            _ => 0,
        })
    })??;
    if trace {
        eprintln!("cuFuncGetAttribute(attr={:?}) -> {}", attrib, value);
    }
    unsafe { *pi = value as i32 };
    Ok(())
}

pub(crate) fn set_attribute(
    func: *mut Function,
    attrib: CUfunction_attribute,
    value: i32,
) -> Result<(), CUresult> {
    if func == ptr::null_mut() || value < 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock_function(func, |func| -> Result<(), CUresult> {
        match attrib {
            // The L0 driver sizes dynamic shared memory per launch (the
            // sharedMemBytes argument of cuLaunchKernel), so there is no
            // per-function state to persist. The remaining attributes are
            // advisory (cluster dims, cache prefs, scheduling policy) and are
            // safe to accept-and-ignore on in-order queues; rejecting them
            // breaks runtimes that set them unconditionally.
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            | CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => {
                Ok(())
            }
            _ => Ok(())
        }
    })??;
    Ok(())
}

pub(crate) fn set_block_shape(func: *mut Function, x: i32, y: i32, z: i32) -> Result<(), CUresult> {
    if func == ptr::null_mut() || x < 0 || y < 0 || z < 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock_function(func, |func| {
        func.legacy_args.block_shape = Some((x, y, z));
    })
}

use crate::ast;
use half::f16;
use rspirv::dr;
use std::cell::RefCell;
use std::collections::{hash_map, HashMap, HashSet};
use std::{borrow::Cow, collections::BTreeSet, ffi::CString, hash::Hash, iter, mem, rc::Rc};

use rspirv::binary::Assemble;

static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../lib/zluda_ptx_impl.spv");
static ZLUDA_PTX_PREFIX: &'static str = "__zluda_ptx_impl__";

quick_error! {
    #[derive(Debug)]
    pub enum TranslateError {
        UnknownSymbol {}
        /// Same as UnknownSymbol, but names the offending identifier.
        UnknownSymbolName(name: String) {
            display("unknown symbol: {}", name)
        }
        UntypedSymbol {}
        /// A symbol without a type; names the offending identifier.
        UntypedSymbolName(name: String) {
            display("untyped symbol: {}", name)
        }
        MismatchedType {}
        /// The same, carrying the source location that produced it.
        MismatchedTypeAt(location: String) {
            display("type mismatch at {}", location)
        }
        /// A type mismatch, with the instruction that caused it.
        MismatchedTypeIn(op: String) {
            display("type mismatch in {}", op)
        }
        Spirv(err: rspirv::dr::Error) {
            from()
            display("{}", err)
            cause(err)
        }
        Unreachable {}
        /// An unreachable branch was hit; carries the source location.
        UnreachableAt(location: String) {
            display("unreachable at {}", location)
        }
        Todo {}
        /// The same, carrying the source location of the unimplemented path.
        TodoAt(location: String) {
            display("unimplemented at {}", location)
        }
    }
}

/// A type mismatch, with the place it was detected.
#[track_caller]
fn error_mismatched_type() -> TranslateError {
    let location = std::panic::Location::caller();
    TranslateError::MismatchedTypeAt(format!("{}:{}", location.file(), location.line()))
}

/// A path that is not implemented, with the place it was reached from.
#[track_caller]
fn error_todo() -> TranslateError {
    let location = std::panic::Location::caller();
    TranslateError::TodoAt(format!("{}:{}", location.file(), location.line()))
}

#[cfg(debug_assertions)]
fn error_unreachable() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
#[track_caller]
fn error_unreachable() -> TranslateError {
    // The location tells which code path hit the unreachable branch.
    let location = std::panic::Location::caller();
    TranslateError::UnreachableAt(format!("{}:{}", location.file(), location.line()))
}

#[derive(PartialEq, Eq, Hash, Clone)]
enum SpirvType {
    Base(SpirvScalarKey),
    Vector(SpirvScalarKey, u8),
    Array(SpirvScalarKey, Vec<u32>),
    Pointer(Box<SpirvType>, spirv::StorageClass),
    Func(Option<Box<SpirvType>>, Vec<SpirvType>),
    Struct(Vec<SpirvScalarKey>),
}

impl SpirvType {
    fn new(t: ast::Type) -> Self {
        match t {
            // A packed half (".f16x2") is a vector of two 16-bit floats, so
            // the scalar mapping has to go through the type conversion that
            // knows about it.
            ast::Type::Scalar(t) => SpirvType::from(t),
            ast::Type::Vector(typ, len) => SpirvType::Vector(typ.into(), len),
            ast::Type::Array(t, len) => SpirvType::Array(t.into(), len),
            ast::Type::Pointer(pointer_t, space) => SpirvType::Pointer(
                Box::new(SpirvType::Base(pointer_t.into())),
                space.to_spirv(),
            ),
            ast::Type::Struct(members) => {
                SpirvType::Struct(members.into_iter().map(Into::into).collect())
            }
        }
    }

    fn pointer_to(t: ast::Type, outer_space: spirv::StorageClass) -> Self {
        let key = Self::new(t);
        SpirvType::Pointer(Box::new(key), outer_space)
    }
}

impl From<ast::ScalarType> for SpirvType {
    fn from(t: ast::ScalarType) -> Self {
        match t {
            // A packed half (".f16x2") is a vector of two 16-bit floats.
            ast::ScalarType::F16x2 => {
                SpirvType::Vector(SpirvScalarKey::F16, 2)
            }
            _ => SpirvType::Base(t.into()),
        }
    }
}

struct TypeWordMap {
    void: spirv::Word,
    complex: HashMap<SpirvType, spirv::Word>,
    constants: HashMap<(SpirvType, u64), spirv::Word>,
}

// SPIR-V integer type definitions are signless, more below:
// https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_unsignedsigned_a_unsigned_versus_signed_integers
// https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_validation_rules_for_kernel_a_href_capability_capabilities_a
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum SpirvScalarKey {
    B8,
    B16,
    B32,
    B64,
    F16,
    F32,
    F64,
    Pred,
    F16x2,
}

impl From<ast::ScalarType> for SpirvScalarKey {
    fn from(t: ast::ScalarType) -> Self {
        match t {
            ast::ScalarType::B8 | ast::ScalarType::U8 | ast::ScalarType::S8 => SpirvScalarKey::B8,
            ast::ScalarType::B16 | ast::ScalarType::U16 | ast::ScalarType::S16 => {
                SpirvScalarKey::B16
            }
            ast::ScalarType::B32 | ast::ScalarType::U32 | ast::ScalarType::S32 => {
                SpirvScalarKey::B32
            }
            ast::ScalarType::B64 | ast::ScalarType::U64 | ast::ScalarType::S64 => {
                SpirvScalarKey::B64
            }
            ast::ScalarType::F16 => SpirvScalarKey::F16,
            ast::ScalarType::F32 => SpirvScalarKey::F32,
            ast::ScalarType::F64 => SpirvScalarKey::F64,
            // A packed half is a vector of two 16-bit floats; the element
            // type is used when a vector of them is built.
            ast::ScalarType::F16x2 => SpirvScalarKey::F16,
            ast::ScalarType::Pred => SpirvScalarKey::Pred,
        }
    }
}

impl TypeWordMap {
    fn new(b: &mut dr::Builder) -> TypeWordMap {
        let void = b.type_void(None);
        TypeWordMap {
            void: void,
            complex: HashMap::<SpirvType, spirv::Word>::new(),
            constants: HashMap::new(),
        }
    }

    fn void(&self) -> spirv::Word {
        self.void
    }

    fn get_or_add_scalar(&mut self, b: &mut dr::Builder, t: ast::ScalarType) -> spirv::Word {
        let key: SpirvScalarKey = t.into();
        self.get_or_add_spirv_scalar(b, key)
    }

    fn get_or_add_spirv_scalar(&mut self, b: &mut dr::Builder, key: SpirvScalarKey) -> spirv::Word {
        *self
            .complex
            .entry(SpirvType::Base(key))
            .or_insert_with(|| match key {
                SpirvScalarKey::B8 => b.type_int(None, 8, 0),
                SpirvScalarKey::B16 => b.type_int(None, 16, 0),
                SpirvScalarKey::B32 => b.type_int(None, 32, 0),
                SpirvScalarKey::B64 => b.type_int(None, 64, 0),
                SpirvScalarKey::F16 => b.type_float(None, 16),
                SpirvScalarKey::F32 => b.type_float(None, 32),
                SpirvScalarKey::F64 => b.type_float(None, 64),
                SpirvScalarKey::Pred => b.type_bool(None),
                SpirvScalarKey::F16x2 => todo!(),
            })
    }

    fn get_or_add(&mut self, b: &mut dr::Builder, t: SpirvType) -> spirv::Word {
        match t {
            SpirvType::Base(key) => self.get_or_add_spirv_scalar(b, key),
            SpirvType::Pointer(ref typ, storage) => {
                let base = self.get_or_add(b, *typ.clone());
                *self
                    .complex
                    .entry(t)
                    .or_insert_with(|| b.type_pointer(None, storage, base))
            }
            SpirvType::Vector(typ, len) => {
                let base = self.get_or_add_spirv_scalar(b, typ);
                *self
                    .complex
                    .entry(t)
                    .or_insert_with(|| b.type_vector(None, base, len as u32))
            }
            SpirvType::Array(typ, array_dimensions) => {
                let (base_type, length) = match &*array_dimensions {
                    &[] => {
                        return self.get_or_add(b, SpirvType::Base(typ));
                    }
                    &[len] => {
                        let u32_type = self.get_or_add_scalar(b, ast::ScalarType::U32);
                        let base = self.get_or_add_spirv_scalar(b, typ);
                        let len_const = b.constant_u32(u32_type, None, len);
                        (base, len_const)
                    }
                    array_dimensions => {
                        let u32_type = self.get_or_add_scalar(b, ast::ScalarType::U32);
                        let base = self
                            .get_or_add(b, SpirvType::Array(typ, array_dimensions[1..].to_vec()));
                        let len_const = b.constant_u32(u32_type, None, array_dimensions[0]);
                        (base, len_const)
                    }
                };
                *self
                    .complex
                    .entry(SpirvType::Array(typ, array_dimensions))
                    .or_insert_with(|| b.type_array(None, base_type, length))
            }
            SpirvType::Func(ref out_params, ref in_params) => {
                let out_t = match out_params {
                    Some(p) => self.get_or_add(b, *p.clone()),
                    None => self.void(),
                };
                let in_t = in_params
                    .iter()
                    .map(|t| self.get_or_add(b, t.clone()))
                    .collect::<Vec<_>>();
                *self
                    .complex
                    .entry(t)
                    .or_insert_with(|| b.type_function(None, out_t, in_t))
            }
            SpirvType::Struct(ref underlying) => {
                let underlying_ids = underlying
                    .iter()
                    .map(|t| self.get_or_add_spirv_scalar(b, *t))
                    .collect::<Vec<_>>();
                *self
                    .complex
                    .entry(t)
                    .or_insert_with(|| b.type_struct(None, underlying_ids))
            }
        }
    }

    fn get_or_add_fn(
        &mut self,
        b: &mut dr::Builder,
        in_params: impl Iterator<Item = SpirvType>,
        mut out_params: impl ExactSizeIterator<Item = SpirvType>,
    ) -> (spirv::Word, spirv::Word) {
        let (out_args, out_spirv_type) = if out_params.len() == 0 {
            (None, self.void())
        } else if out_params.len() == 1 {
            let arg_as_key = out_params.next().unwrap();
            (
                Some(Box::new(arg_as_key.clone())),
                self.get_or_add(b, arg_as_key),
            )
        } else {
            // TODO: support multiple return values
            todo!()
        };
        (
            out_spirv_type,
            self.get_or_add(b, SpirvType::Func(out_args, in_params.collect::<Vec<_>>())),
        )
    }

    fn get_or_add_constant(
        &mut self,
        b: &mut dr::Builder,
        typ: &ast::Type,
        init: &[u8],
    ) -> Result<spirv::Word, TranslateError> {
        Ok(match typ {
            // A parameter struct never has a literal initialiser.
            ast::Type::Struct(_) => return Err(error_todo()),
            ast::Type::Scalar(t) => match t {
                ast::ScalarType::B8 | ast::ScalarType::U8 | ast::ScalarType::S8 => self
                    .get_or_add_constant_single::<u8, _, _>(
                        b,
                        *t,
                        init,
                        |v| v as u64,
                        |b, result_type, v| b.constant_u32(result_type, None, v as u32),
                    ),
                ast::ScalarType::B16 | ast::ScalarType::U16 | ast::ScalarType::S16 => self
                    .get_or_add_constant_single::<u16, _, _>(
                        b,
                        *t,
                        init,
                        |v| v as u64,
                        |b, result_type, v| b.constant_u32(result_type, None, v as u32),
                    ),
                ast::ScalarType::B32 | ast::ScalarType::U32 | ast::ScalarType::S32 => self
                    .get_or_add_constant_single::<u32, _, _>(
                        b,
                        *t,
                        init,
                        |v| v as u64,
                        |b, result_type, v| b.constant_u32(result_type, None, v),
                    ),
                ast::ScalarType::B64 | ast::ScalarType::U64 | ast::ScalarType::S64 => self
                    .get_or_add_constant_single::<u64, _, _>(
                        b,
                        *t,
                        init,
                        |v| v,
                        |b, result_type, v| b.constant_u64(result_type, None, v),
                    ),
                ast::ScalarType::F16 => self.get_or_add_constant_single::<f16, _, _>(
                    b,
                    *t,
                    init,
                    |v| unsafe { mem::transmute::<_, u16>(v) } as u64,
                    |b, result_type, v| b.constant_f32(result_type, None, v.to_f32()),
                ),
                ast::ScalarType::F32 => self.get_or_add_constant_single::<f32, _, _>(
                    b,
                    *t,
                    init,
                    |v| unsafe { mem::transmute::<_, u32>(v) } as u64,
                    |b, result_type, v| b.constant_f32(result_type, None, v),
                ),
                ast::ScalarType::F64 => self.get_or_add_constant_single::<f64, _, _>(
                    b,
                    *t,
                    init,
                    |v| unsafe { mem::transmute::<_, u64>(v) },
                    |b, result_type, v| b.constant_f64(result_type, None, v),
                ),
                ast::ScalarType::F16x2 => return Err(error_todo()),
                ast::ScalarType::Pred => self.get_or_add_constant_single::<u8, _, _>(
                    b,
                    *t,
                    init,
                    |v| v as u64,
                    |b, result_type, v| {
                        if v == 0 {
                            b.constant_false(result_type, None)
                        } else {
                            b.constant_true(result_type, None)
                        }
                    },
                ),
            },
            ast::Type::Vector(typ, len) => {
                let result_type =
                    self.get_or_add(b, SpirvType::Vector(SpirvScalarKey::from(*typ), *len));
                let size_of_t = typ.size_of();
                let components = (0..*len)
                    .map(|x| {
                        self.get_or_add_constant(
                            b,
                            &ast::Type::Scalar(*typ),
                            &init[((size_of_t as usize) * (x as usize))..],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                b.constant_composite(result_type, None, components.into_iter())
            }
            ast::Type::Array(typ, dims) => match dims.as_slice() {
                [] => return Err(error_unreachable()),
                [dim] => {
                    // One SPIR-V instruction holds at most 65535 words, and a
                    // composite spends one word per constituent, so a larger
                    // array cannot be written flat. Naming the size here beats
                    // emitting a module that the validator rejects.
                    if *dim as usize >= 65535 - 2 {
                        return Err(TranslateError::MismatchedTypeIn(format!(
                            "an initialiser of {} elements is too large for one SPIR-V instruction",
                            dim
                        )));
                    }
                    let result_type = self
                        .get_or_add(b, SpirvType::Array(SpirvScalarKey::from(*typ), vec![*dim]));
                    let size_of_t = typ.size_of();
                    let components = (0..*dim)
                        .map(|x| {
                            self.get_or_add_constant(
                                b,
                                &ast::Type::Scalar(*typ),
                                &init[((size_of_t as usize) * (x as usize))..],
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    b.constant_composite(result_type, None, components.into_iter())
                }
                [first_dim, rest @ ..] => {
                    let result_type = self.get_or_add(
                        b,
                        SpirvType::Array(SpirvScalarKey::from(*typ), rest.to_vec()),
                    );
                    let size_of_t = rest
                        .iter()
                        .fold(typ.size_of() as u32, |x, y| (x as u32) * (*y));
                    let components = (0..*first_dim)
                        .map(|x| {
                            self.get_or_add_constant(
                                b,
                                &ast::Type::Array(*typ, rest.to_vec()),
                                &init[((size_of_t as usize) * (x as usize))..],
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    b.constant_composite(result_type, None, components.into_iter())
                }
            },
            ast::Type::Pointer(..) => return Err(error_unreachable()),
        })
    }

    fn get_or_add_constant_single<
        T: Copy,
        CastAsU64: FnOnce(T) -> u64,
        InsertConstant: FnOnce(&mut dr::Builder, spirv::Word, T) -> spirv::Word,
    >(
        &mut self,
        b: &mut dr::Builder,
        key: ast::ScalarType,
        init: &[u8],
        cast: CastAsU64,
        f: InsertConstant,
    ) -> spirv::Word {
        let value = unsafe { *(init.as_ptr() as *const T) };
        let value_64 = cast(value);
        let ht_key = (SpirvType::Base(SpirvScalarKey::from(key)), value_64);
        match self.constants.get(&ht_key) {
            Some(value) => *value,
            None => {
                let spirv_type = self.get_or_add_scalar(b, key);
                let result = f(b, spirv_type, value);
                self.constants.insert(ht_key, result);
                result
            }
        }
    }
}

pub struct Module {
    pub spirv: dr::Module,
    pub kernel_info: HashMap<String, KernelInfo>,
    pub should_link_ptx_impl: Option<&'static [u8]>,
    pub build_options: CString,
}
impl Module {
    pub fn assemble(&self) -> Vec<u32> {
        self.spirv.assemble()
    }
}

#[derive(Clone)]
pub struct KernelInfo {
    pub arguments_sizes: Vec<usize>,
    pub uses_shared_mem: bool,
    /// Bytes to ask for on top of the application's dynamic shared request.
    pub shared_mem_prefix: usize,
}

pub fn to_spirv_module<'a>(ast: ast::Module<'a>) -> Result<Module, TranslateError> {
    let mut id_defs = GlobalStringIdResolver::new(1);
    let mut ptx_impl_imports = HashMap::new();
    let directives = ast
        .directives
        .into_iter()
        .filter_map(|directive| {
            translate_directive(&mut id_defs, &mut ptx_impl_imports, directive).transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let must_link_ptx_impl = ptx_impl_imports.len() > 0;
    let directives = ptx_impl_imports
        .into_iter()
        .map(|(_, v)| v)
        .chain(directives.into_iter())
        .collect::<Vec<_>>();
    let mut builder = dr::Builder::new();
    builder.reserve_ids(id_defs.current_id());
    let call_map = get_kernels_call_map(&directives);
    let callee_param_types = get_callee_param_types(&directives);
    let mut directives = convert_dynamic_shared_memory_usage(directives, &mut || builder.id());
    directives = pack_large_byte_globals(directives);
    normalize_variable_decls(&mut directives);
    let denorm_information = compute_denorm_information(&directives);
    // https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_logicallayout_a_logical_layout_of_a_module
    builder.set_version(1, 3);
    emit_capabilities(&mut builder);
    emit_extensions(&mut builder);
    let opencl_id = emit_opencl_import(&mut builder);
    emit_memory_model(&mut builder);
    let mut map = TypeWordMap::new(&mut builder);
    // ldmatrix selects the thread that holds a matrix row by its lane id, so
    // the builtin is needed even when the kernel never reads %laneid.
    let needs_laneid = directives.iter().any(|directive| match directive {
        Directive::Method(method) => match &method.body {
            Some(body) => statements_use_laneid(body),
            None => false,
        },
        _ => false,
    });
    if needs_laneid {
        id_defs
            .special_registers
            .get_or_add(&mut id_defs.current_id, PtxSpecialRegister::Laneid);
    }
    emit_builtins(&mut builder, &mut map, &id_defs);
    let mut kernel_info = HashMap::new();
    let build_options = emit_denorm_build_string(&call_map, &denorm_information);
    emit_directives(
        &mut builder,
        &mut map,
        &id_defs,
        opencl_id,
        &denorm_information,
        &call_map,
        &callee_param_types,
        directives,
        &mut kernel_info,
    )?;
    // A float atomic is an extension instruction, and a module that uses one has
    // to declare both the capability and the extension. Declared only when used,
    // because an unused capability in every module is a larger change than it
    // looks: torch's large reductions are the path that needs it.
    let uses_float_atomic = builder
        .module_mut()
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter())
        .flat_map(|block| block.instructions.iter())
        .any(|instruction| instruction.class.opcode == spirv::Op::AtomicFAddEXT);
    if uses_float_atomic {
        builder.capability(spirv::Capability::AtomicFloat32AddEXT);
        builder.extension("SPV_EXT_shader_atomic_float_add");
    }
    let spirv = builder.module();
    Ok(Module {
        spirv,
        kernel_info,
        should_link_ptx_impl: if must_link_ptx_impl {
            Some(ZLUDA_PTX_IMPL)
        } else {
            None
        },
        build_options,
    })
}

// TODO: remove this once we have perf-function support for denorms
fn emit_denorm_build_string<'input>(
    call_map: &HashMap<&str, HashSet<u32>>,
    denorm_information: &HashMap<
        ast::MethodName<'input, spirv::Word>,
        HashMap<u8, (spirv::FPDenormMode, isize)>,
    >,
) -> CString {
    let denorm_counts = denorm_information
        .iter()
        .map(|(method, meth_denorm)| {
            let f16_count = meth_denorm
                .get(&(mem::size_of::<f16>() as u8))
                .unwrap_or(&(spirv::FPDenormMode::FlushToZero, 0))
                .1;
            let f32_count = meth_denorm
                .get(&(mem::size_of::<f32>() as u8))
                .unwrap_or(&(spirv::FPDenormMode::FlushToZero, 0))
                .1;
            (method, (f16_count + f32_count))
        })
        .collect::<HashMap<_, _>>();
    let mut flush_over_preserve = 0;
    for (kernel, children) in call_map {
        flush_over_preserve += *denorm_counts
            .get(&ast::MethodName::Kernel(kernel))
            .unwrap_or(&0);
        for child_fn in children {
            flush_over_preserve += *denorm_counts
                .get(&ast::MethodName::Func(*child_fn))
                .unwrap_or(&0);
        }
    }
    // The optimizer stays on: without it a big kernel spills past the
    // scratch space limit and its whole unit fails to build.
    // ZLUDA_PTX_NO_OPT=1 turns it off.
    let optimize = std::env::var_os("ZLUDA_PTX_NO_OPT").is_none();
    let opt_flag = if optimize { "" } else { " -cl-opt-disable" };
    let denorm_flag = if flush_over_preserve > 0 {
        " -ze-denorms-are-zero"
    } else {
        ""
    };
    CString::new(format!(
        "-ze-take-global-address{}{}",
        denorm_flag, opt_flag
    ))
    .unwrap()
}

/// Collect the ids of the functions called from a function body.
fn collect_called_functions(body: &[ExpandedStatement]) -> HashSet<spirv::Word> {
    let mut result = HashSet::new();
    for statement in body {
        match statement {
            Statement::Call(call) => {
                result.insert(call.name);
            }
            _ => {}
        }
    }
    result
}

/// Emit an OpFunction declaration (no body) for a function that is only
/// declared (`.extern .func`): a call to it still needs a target id.
fn emit_function_declaration<'input>(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    f: &Function<'input>,
) -> Result<(), TranslateError> {
    let func_decl = (*f.func_decl).borrow();
    let fn_id = match func_decl.name {
        ast::MethodName::Func(name) => name,
        // A kernel without a body cannot happen; skip it.
        ast::MethodName::Kernel(_) => return Ok(()),
    };
    let (ret_type, func_type) = get_function_type(
        builder,
        map,
        func_decl.effective_input_arguments().map(|(_, typ)| typ),
        &func_decl.return_arguments,
    );
    builder.begin_function(
        ret_type,
        Some(fn_id),
        spirv::FunctionControl::NONE,
        func_type,
    )?;
    for (name, typ) in func_decl.effective_input_arguments() {
        let result_type = map.get_or_add(builder, typ);
        builder.function_parameter(Some(name), result_type)?;
    }
    if f.import_as.is_some() {
        // A function provided by another module (the zluda PTX implementation
        // library) must carry the Import linkage, or spirv-val rejects the
        // declaration.
        builder.end_function()?;
        let linkage_name = f.import_as.as_ref().unwrap().clone();
        builder.decorate(
            fn_id,
            spirv::Decoration::LinkageAttributes,
            [
                dr::Operand::LiteralString(linkage_name),
                dr::Operand::LinkageType(spirv::LinkageType::Import),
            ]
            .iter()
            .cloned(),
        );
        return Ok(());
    }
    // Called but never defined in this module: CUDA resolves those at device
    // link time, but each module is loaded on its own here, so the call gets a
    // stub body (the call usually sits in a branch that is never taken).
    builder.begin_block(None)?;
    if func_decl.return_arguments.is_empty() {
        builder.ret()?;
    } else {
        let ret = func_decl.return_arguments[0].v_type.clone();
        let ret_type = map.get_or_add(builder, SpirvType::new(ret.clone()));
        let zero = match ret {
            ast::Type::Scalar(ast::ScalarType::F32) => builder.constant_f32(ret_type, None, 0.0),
            ast::Type::Scalar(ast::ScalarType::F64) => builder.constant_f64(ret_type, None, 0.0),
            ast::Type::Scalar(ast::ScalarType::F16) => builder.constant_f32(ret_type, None, 0.0),
            _ => builder.constant_u32(ret_type, None, 0),
        };
        builder.ret_value(zero)?;
    }
    builder.end_function()?;
    Ok(())
}

fn emit_directives<'input>(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    id_defs: &GlobalStringIdResolver<'input>,
    opencl_id: spirv::Word,
    denorm_information: &HashMap<
        ast::MethodName<'input, spirv::Word>,
        HashMap<u8, (spirv::FPDenormMode, isize)>,
    >,
    call_map: &HashMap<&'input str, HashSet<spirv::Word>>,
    callee_param_types: &HashMap<spirv::Word, Vec<SpirvType>>,
    directives: Vec<Directive<'input>>,
    kernel_info: &mut HashMap<String, KernelInfo>,
) -> Result<(), TranslateError> {
    let empty_body = Vec::new();
    // The functions defined (with a body) in this module.
    let defined_functions: HashSet<spirv::Word> = directives
        .iter()
        .filter_map(|d| match d {
            Directive::Method(f) if f.body.is_some() => match (*f.func_decl).borrow().name {
                ast::MethodName::Func(name) => Some(name),
                ast::MethodName::Kernel(_) => None,
            },
            _ => None,
        })
        .collect();
    // The functions that are actually called. A declaration that is never
    // called must not be emitted: its Import linkage would make the module
    // unlinkable.
    let called_functions: HashSet<spirv::Word> = directives
        .iter()
        .filter_map(|d| match d {
            Directive::Method(f) => f.body.as_ref().map(|body| collect_called_functions(body)),
            _ => None,
        })
        .flatten()
        .collect();
    for d in directives.iter() {
        match d {
            Directive::Variable(_, var) => {
                emit_variable(builder, map, &var)?;
            }
            Directive::Method(f) => {
                let f_body = match &f.body {
                    Some(f) => f,
                    None => {
                        if f.import_as.is_some() {
                            &empty_body
                        } else {
                            // A bodyless function is only declared, and a call
                            // still needs a target id, so emit an OpFunction
                            // without a body. A function that is declared and
                            // defined later in the module needs no declaration.
                            let fn_id = match (*f.func_decl).borrow().name {
                                ast::MethodName::Func(name) => Some(name),
                                ast::MethodName::Kernel(_) => None,
                            };
                            if let Some(fn_id) = fn_id {
                                if !defined_functions.contains(&fn_id)
                                    && called_functions.contains(&fn_id)
                                {
                                    emit_function_declaration(builder, map, f)?;
                                }
                            }
                            continue;
                        }
                    }
                };
                for var in f.globals.iter() {
                    emit_variable(builder, map, var)?;
                }
                let func_decl = (*f.func_decl).borrow();
                let fn_id = emit_function_header(
                    builder,
                    map,
                    &id_defs,
                    &f.globals,
                    &*func_decl,
                    &denorm_information,
                    call_map,
                    &directives,
                    kernel_info,
                )?;
                let mut emitted_execution_mode = false;
                if let ast::MethodName::Kernel(_) = func_decl.name {
                    for t in f.tuning.iter() {
                        match *t {
                            ast::TuningDirective::MaxNtid(nx, ny, nz) => {
                                // .maxntid caps the TOTAL thread count (x*y*z);
                                // MaxWorkgroupSizeINTEL is per dimension, so
                                // allow up to the total in each.
                                let total = (nx * ny * nz).clamp(1, 256);
                                builder.execution_mode(
                                    fn_id,
                                    spirv_headers::ExecutionMode::MaxWorkgroupSizeINTEL,
                                    [total, total, total],
                                );
                                emitted_execution_mode = true;
                            }
                            ast::TuningDirective::ReqNtid(nx, ny, nz) => {
                                builder.execution_mode(
                                    fn_id,
                                    spirv_headers::ExecutionMode::LocalSize,
                                    [nx, ny, nz],
                                );
                                emitted_execution_mode = true;
                            }
                            // Too architecture specific
                            ast::TuningDirective::MaxNReg(..)
                            | ast::TuningDirective::MinNCtaPerSm(..)
                            | ast::TuningDirective::NoReturn => {}
                        }
                    }
                    if !emitted_execution_mode {
                        // No tuning directive: don't force a LocalSize (that
                        // would cap the launch block). Allow up to 256 per
                        // dimension instead.
                        builder.execution_mode(
                            fn_id,
                            spirv_headers::ExecutionMode::MaxWorkgroupSizeINTEL,
                            [256, 256, 256],
                        );
                    }
                    // A PTX warp is 32 lanes but this GPU's default subgroup is
                    // 16, and a shuffle past lane 15 reads outside it and
                    // returns zero.
                    if statements_need_full_warp(&f_body)
                        || special_registers_need_full_warp(&id_defs.special_registers)
                    {
                        builder.execution_mode(
                            fn_id,
                            spirv_headers::ExecutionMode::SubgroupSize,
                            [32],
                        );
                    }
                }
                let laneid = id_defs
                    .special_registers
                    .builtins()
                    .find(|(sreg, _)| *sreg == PtxSpecialRegister::Laneid)
                    .map(|(_, id)| id);
                emit_function_body_ops(
                    builder,
                    map,
                    opencl_id,
                    &f_body,
                    callee_param_types,
                    laneid,
                )?;
                if builder.selected_block().is_some() {
                    builder.ret()?;
                }
                builder.end_function()?;
                let fn_idx = builder.module_ref().functions.len() - 1;
                if std::env::var_os("ZLUDA_PTX_DISABLE_RPO").is_none() {
                    reorder_blocks_rpo(builder.module_mut(), fn_idx);
                }
                if let (
                    ast::MethodDeclaration {
                        name: ast::MethodName::Func(fn_id),
                        ..
                    },
                    Some(name),
                ) = (&*func_decl, &f.import_as)
                {
                    builder.decorate(
                        *fn_id,
                        spirv::Decoration::LinkageAttributes,
                        [
                            dr::Operand::LiteralString(name.clone()),
                            dr::Operand::LinkageType(spirv::LinkageType::Import),
                        ]
                        .iter()
                        .cloned(),
                    );
                }
            }
        }
    }
    Ok(())
}

fn get_kernels_call_map<'input>(
    module: &[Directive<'input>],
) -> HashMap<&'input str, HashSet<spirv::Word>> {
    let mut directly_called_by = HashMap::new();
    for directive in module {
        match directive {
            Directive::Method(Function {
                func_decl,
                body: Some(statements),
                ..
            }) => {
                let call_key: ast::MethodName<_> = (**func_decl).borrow().name;
                if let hash_map::Entry::Vacant(entry) = directly_called_by.entry(call_key) {
                    entry.insert(Vec::new());
                }
                for statement in statements {
                    match statement {
                        Statement::Call(call) => {
                            multi_hash_map_append(&mut directly_called_by, call_key, call.name);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let mut result = HashMap::new();
    for (method_key, children) in directly_called_by.iter() {
        match method_key {
            ast::MethodName::Kernel(name) => {
                let mut visited = HashSet::new();
                for child in children {
                    add_call_map_single(&directly_called_by, &mut visited, *child);
                }
                result.insert(*name, visited);
            }
            ast::MethodName::Func(_) => {}
        }
    }
    result
}

// Map from a function id to its effective input argument types, so call
// sites can match the callee's pointer types.
fn get_callee_param_types<'input>(
    module: &[Directive<'input>],
) -> HashMap<spirv::Word, Vec<SpirvType>> {
    let mut result = HashMap::new();
    for directive in module {
        if let Directive::Method(Function { func_decl, .. }) = directive {
            let decl = (*func_decl).borrow();
            let id = match decl.name {
                ast::MethodName::Kernel(_) => continue,
                ast::MethodName::Func(id) => id,
            };
            let types = decl
                .input_arguments
                .iter()
                .map(|arg| {
                    if decl.name.is_kernel() || arg.state_space == ast::StateSpace::Reg {
                        SpirvType::new(arg.v_type.clone())
                    } else {
                        SpirvType::pointer_to(arg.v_type.clone(), arg.state_space.to_spirv())
                    }
                })
                .collect();
            result.insert(id, types);
        }
    }
    result
}

fn add_call_map_single<'input>(
    directly_called_by: &MultiHashMap<ast::MethodName<'input, spirv::Word>, spirv::Word>,
    visited: &mut HashSet<spirv::Word>,
    current: spirv::Word,
) {
    if !visited.insert(current) {
        return;
    }
    if let Some(children) = directly_called_by.get(&ast::MethodName::Func(current)) {
        for child in children {
            add_call_map_single(directly_called_by, visited, *child);
        }
    }
}

type MultiHashMap<K, V> = HashMap<K, Vec<V>>;

fn multi_hash_map_append<K: Eq + std::hash::Hash, V>(m: &mut MultiHashMap<K, V>, key: K, value: V) {
    match m.entry(key) {
        hash_map::Entry::Occupied(mut entry) => {
            entry.get_mut().push(value);
        }
        hash_map::Entry::Vacant(entry) => {
            entry.insert(vec![value]);
        }
    }
}

/*
    PTX represents dynamically allocated shared local memory as
        .extern .shared .b32 shared_mem[];
    In SPIRV/OpenCL world this is expressed as an additional argument
    This pass looks for all uses of .extern .shared and converts them to
    an additional method argument
    The question is how this artificial argument should be expressed. There are
    several options:
    * Straight conversion:
        .shared .b32 shared_mem[]
    * Introduce .param_shared statespace:
        .param_shared .b32 shared_mem
        or
        .param_shared .b32 shared_mem[]
    * Introduce .shared_ptr <SCALAR> type:
        .param .shared_ptr .b32 shared_mem
    * Reuse .ptr hint:
        .param .u64 .ptr shared_mem
      This is the most tempting, but also the most nonsensical, .ptr is just a
      hint, which has no semantical meaning (and the output of our
      transformation has a semantical meaning - we emit additional
      "OpFunctionParameter ..." with type "OpTypePointer Workgroup ...")
*/
/// Rewrite a byte array whose initialiser is too large for one SPIR-V
/// instruction into the same bytes seen as 32-bit words.
///
/// One instruction holds at most 65535 words and a composite spends one word per
/// constituent, so an initialiser above that cannot be written flat. The bytes
/// do not change, only their width, and accesses to these globals already go
/// through a byte pointer, so nothing else has to know.
fn widen_large_byte_array(var: &mut ast::Variable<spirv::Word>, limit: u32) {
    if let ast::Type::Array(ast::ScalarType::B8, dims) = &var.v_type {
        let count: u32 = dims.iter().product();
        if count >= limit {
            // Up to three bytes of padding; each global is its own variable, so
            // growing one cannot disturb another.
            let words = (count + 3) / 4;
            var.array_init.resize(words as usize * 4, 0);
            var.v_type = ast::Type::Array(ast::ScalarType::U32, vec![words]);
        }
    }
}

fn pack_large_byte_globals<'input>(module: Vec<Directive<'input>>) -> Vec<Directive<'input>> {
    const LIMIT: u32 = 65535 - 2;
    module
        .into_iter()
        .map(|directive| match directive {
            Directive::Variable(linking, mut var) => {
                widen_large_byte_array(&mut var, LIMIT);
                Directive::Variable(linking, var)
            }
            // A lifted kernel keeps the data it initialises in its own globals.
            Directive::Method(mut func) => {
                for var in func.globals.iter_mut() {
                    widen_large_byte_array(var, LIMIT);
                }
                Directive::Method(func)
            }
            directive => directive,
        })
        .collect()
}

fn convert_dynamic_shared_memory_usage<'input>(
    module: Vec<Directive<'input>>,
    new_id: &mut impl FnMut() -> spirv::Word,
) -> Vec<Directive<'input>> {
    let mut extern_shared_decls = HashMap::new();
    let mut extern_shared_sizes = HashMap::new();
    for dir in module.iter() {
        match dir {
            Directive::Variable(
                linking,
                ast::Variable {
                    v_type: ast::Type::Array(p_type, dims),
                    state_space: ast::StateSpace::Shared,
                    name,
                    ..
                },
            ) if linking.contains(ast::LinkingDirective::EXTERN) => {
                extern_shared_decls.insert(*name, *p_type);
                // The window is dynamic even when a size is written down: a
                // lifted kernel flattens its static and dynamic shared memory
                // into one symbol, and the size it declares is only the static
                // part of it, which the launch has to ask for on top of the
                // dynamic request. Treating it as a static variable instead
                // silently loses every access past that size.
                let count: u32 = dims.iter().product();
                let declared = count as usize * p_type.size_of() as usize;
                extern_shared_sizes.insert(*name, declared);
            }
            _ => {}
        }
    }
    if extern_shared_decls.len() == 0 {
        return module;
    }
    let mut methods_using_extern_shared = HashSet::new();
    let mut directly_called_by = MultiHashMap::new();
    let module = module
        .into_iter()
        .map(|directive| match directive {
            Directive::Method(Function {
                func_decl,
                globals,
                body: Some(statements),
                import_as,
                tuning,
            }) => {
                let call_key = (*func_decl).borrow().name;
                let statements = statements
                    .into_iter()
                    .map(|statement| match statement {
                        Statement::Call(call) => {
                            multi_hash_map_append(&mut directly_called_by, call.name, call_key);
                            Statement::Call(call)
                        }
                        statement => statement.map_id(&mut |id, _| {
                            if extern_shared_decls.contains_key(&id) {
                                methods_using_extern_shared.insert(call_key);
                            }
                            id
                        }),
                    })
                    .collect();
                Directive::Method(Function {
                    func_decl,
                    globals,
                    body: Some(statements),
                    import_as,
                    tuning,
                })
            }
            directive => directive,
        })
        .collect::<Vec<_>>();
    // If there's a chain `kernel` -> `fn1` -> `fn2`, where only `fn2` uses extern shared,
    // make sure it gets propagated to `fn1` and `kernel`
    get_callers_of_extern_shared(&mut methods_using_extern_shared, &directly_called_by);
    // now visit every method declaration and inject those additional arguments
    module
        .into_iter()
        .map(|directive| match directive {
            Directive::Method(Function {
                func_decl,
                globals,
                body: Some(statements),
                import_as,
                tuning,
            }) => {
                if !methods_using_extern_shared.contains(&(*func_decl).borrow().name) {
                    return Directive::Method(Function {
                        func_decl,
                        globals,
                        body: Some(statements),
                        import_as,
                        tuning,
                    });
                }
                let shared_id_param = new_id();
                {
                    let mut func_decl = (*func_decl).borrow_mut();
                    func_decl.shared_mem = Some(shared_id_param);
                    // The largest declared size is the static prefix the
                    // kernel's own addressing assumes.
                    func_decl.shared_mem_prefix =
                        extern_shared_sizes.values().copied().max().unwrap_or(0);
                }
                let statements = replace_uses_of_shared_memory(
                    new_id,
                    &extern_shared_decls,
                    &mut methods_using_extern_shared,
                    shared_id_param,
                    statements,
                );
                Directive::Method(Function {
                    func_decl,
                    globals,
                    body: Some(statements),
                    import_as,
                    tuning,
                })
            }
            directive => directive,
        })
        .collect::<Vec<_>>()
}

fn replace_uses_of_shared_memory<'a>(
    new_id: &mut impl FnMut() -> spirv::Word,
    extern_shared_decls: &HashMap<spirv::Word, ast::ScalarType>,
    methods_using_extern_shared: &mut HashSet<ast::MethodName<'a, spirv::Word>>,
    shared_id_param: spirv::Word,
    statements: Vec<ExpandedStatement>,
) -> Vec<ExpandedStatement> {
    let mut result = Vec::with_capacity(statements.len());
    for statement in statements {
        match statement {
            Statement::Call(mut call) => {
                // We can safely skip checking call arguments,
                // because there's simply no way to pass shared ptr
                // without converting it to .b64 first
                if methods_using_extern_shared.contains(&ast::MethodName::Func(call.name)) {
                    call.input_arguments.push((
                        shared_id_param,
                        ast::Type::Scalar(ast::ScalarType::B8),
                        ast::StateSpace::Shared,
                    ));
                }
                result.push(Statement::Call(call))
            }
            statement => {
                let new_statement = statement.map_id(&mut |id, _| {
                    if let Some(scalar_type) = extern_shared_decls.get(&id) {
                        if *scalar_type == ast::ScalarType::B8 {
                            return shared_id_param;
                        }
                        let replacement_id = new_id();
                        result.push(Statement::Conversion(ImplicitConversion {
                            src: shared_id_param,
                            dst: replacement_id,
                            from_type: ast::Type::Scalar(ast::ScalarType::B8),
                            from_space: ast::StateSpace::Shared,
                            to_type: ast::Type::Scalar(*scalar_type),
                            to_space: ast::StateSpace::Shared,
                            kind: ConversionKind::PtrToPtr,
                        }));
                        replacement_id
                    } else {
                        id
                    }
                });
                result.push(new_statement);
            }
        }
    }
    result
}

fn get_callers_of_extern_shared<'a>(
    methods_using_extern_shared: &mut HashSet<ast::MethodName<'a, spirv::Word>>,
    directly_called_by: &MultiHashMap<spirv::Word, ast::MethodName<'a, spirv::Word>>,
) {
    let direct_uses_of_extern_shared = methods_using_extern_shared
        .iter()
        .filter_map(|method| {
            if let ast::MethodName::Func(f_id) = method {
                Some(*f_id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for fn_id in direct_uses_of_extern_shared {
        get_callers_of_extern_shared_single(methods_using_extern_shared, directly_called_by, fn_id);
    }
}

fn get_callers_of_extern_shared_single<'a>(
    methods_using_extern_shared: &mut HashSet<ast::MethodName<'a, spirv::Word>>,
    directly_called_by: &MultiHashMap<spirv::Word, ast::MethodName<'a, spirv::Word>>,
    fn_id: spirv::Word,
) {
    if let Some(callers) = directly_called_by.get(&fn_id) {
        for caller in callers {
            if methods_using_extern_shared.insert(*caller) {
                if let ast::MethodName::Func(caller_fn) = caller {
                    get_callers_of_extern_shared_single(
                        methods_using_extern_shared,
                        directly_called_by,
                        *caller_fn,
                    );
                }
            }
        }
    }
}

type DenormCountMap<T> = HashMap<T, isize>;

fn denorm_count_map_update<T: Eq + Hash>(map: &mut DenormCountMap<T>, key: T, value: bool) {
    let num_value = if value { 1 } else { -1 };
    denorm_count_map_update_impl(map, key, num_value);
}

fn denorm_count_map_update_impl<T: Eq + Hash>(
    map: &mut DenormCountMap<T>,
    key: T,
    num_value: isize,
) {
    match map.entry(key) {
        hash_map::Entry::Occupied(mut counter) => {
            *(counter.get_mut()) += num_value;
        }
        hash_map::Entry::Vacant(entry) => {
            entry.insert(num_value);
        }
    }
}

// PTX carries per-instruction .ftz; SPIR-V only has a kernel-wide mode, so
// the mode is guessed from which the kernel's float operations use more.
fn compute_denorm_information<'input>(
    module: &[Directive<'input>],
) -> HashMap<ast::MethodName<'input, spirv::Word>, HashMap<u8, (spirv::FPDenormMode, isize)>> {
    let mut denorm_methods = HashMap::new();
    for directive in module {
        match directive {
            Directive::Variable(..) | Directive::Method(Function { body: None, .. }) => {}
            Directive::Method(Function {
                func_decl,
                body: Some(statements),
                ..
            }) => {
                let mut flush_counter = DenormCountMap::new();
                let method_key = (**func_decl).borrow().name;
                for statement in statements {
                    match statement {
                        Statement::Instruction(inst) => {
                            if let Some((flush, width)) = inst.flush_to_zero() {
                                denorm_count_map_update(&mut flush_counter, width, flush);
                            }
                        }
                        Statement::LoadVar(..) => {}
                        Statement::StoreVar(..) => {}
                        Statement::Call(_) => {}
                        Statement::Conditional(_) => {}
                        Statement::Conversion(_) => {}
                        Statement::Constant(_) => {}
                        Statement::RetValue(_, _) => {}
                        Statement::Label(_) => {}
                        Statement::Variable(_) => {}
                        Statement::PtrAccess { .. } => {}
                        Statement::RepackVector(_) => {}
                    }
                }
                denorm_methods.insert(method_key, flush_counter);
            }
        }
    }
    denorm_methods
        .into_iter()
        .map(|(name, v)| {
            let width_to_denorm = v
                .into_iter()
                .map(|(k, flush_over_preserve)| {
                    let mode = if flush_over_preserve > 0 {
                        spirv::FPDenormMode::FlushToZero
                    } else {
                        spirv::FPDenormMode::Preserve
                    };
                    (k, (mode, flush_over_preserve))
                })
                .collect();
            (name, width_to_denorm)
        })
        .collect()
}

fn emit_builtins(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    id_defs: &GlobalStringIdResolver,
) {
    for (reg, id) in id_defs.special_registers.builtins() {
        let result_type = map.get_or_add(
            builder,
            SpirvType::pointer_to(reg.get_type(), spirv::StorageClass::Input),
        );
        builder.variable(result_type, Some(id), spirv::StorageClass::Input, None);
        builder.decorate(
            id,
            spirv::Decoration::BuiltIn,
            [dr::Operand::BuiltIn(reg.get_builtin())].iter().cloned(),
        );
    }
}

fn emit_function_header<'a>(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    defined_globals: &GlobalStringIdResolver<'a>,
    synthetic_globals: &[ast::Variable<spirv::Word>],
    func_decl: &ast::MethodDeclaration<'a, spirv::Word>,
    _denorm_information: &HashMap<
        ast::MethodName<'a, spirv::Word>,
        HashMap<u8, (spirv::FPDenormMode, isize)>,
    >,
    call_map: &HashMap<&'a str, HashSet<spirv::Word>>,
    direcitves: &[Directive],
    kernel_info: &mut HashMap<String, KernelInfo>,
) -> Result<spirv::Word, TranslateError> {
    if let ast::MethodName::Kernel(name) = func_decl.name {
        let args_lens = func_decl
            .input_arguments
            .iter()
            .map(|param| param.v_type.size_of())
            .collect();
        kernel_info.insert(
            name.to_string(),
            KernelInfo {
                arguments_sizes: args_lens,
                uses_shared_mem: func_decl.shared_mem.is_some(),
                shared_mem_prefix: func_decl.shared_mem_prefix,
            },
        );
    }
    let (ret_type, func_type) = get_function_type(
        builder,
        map,
        func_decl.effective_input_arguments().map(|(_, typ)| typ),
        &func_decl.return_arguments,
    );
    let fn_id = match func_decl.name {
        ast::MethodName::Kernel(name) => {
            let fn_id = defined_globals.get_id(name)?;
            let mut global_variables = defined_globals
                .variables_type_check
                .iter()
                .filter_map(|(k, t)| t.as_ref().map(|_| *k))
                .collect::<Vec<_>>();
            let mut interface = defined_globals.special_registers.interface();
            for ast::Variable { name, .. } in synthetic_globals {
                interface.push(*name);
            }
            let empty_hash_set = HashSet::new();
            let child_fns = call_map.get(name).unwrap_or(&empty_hash_set);
            for directive in direcitves {
                match directive {
                    Directive::Method(Function {
                        func_decl, globals, ..
                    }) => {
                        match (**func_decl).borrow().name {
                            ast::MethodName::Func(name) => {
                                if child_fns.contains(&name) {
                                    for var in globals {
                                        interface.push(var.name);
                                    }
                                }
                            }
                            ast::MethodName::Kernel(_) => {}
                        };
                    }
                    _ => {}
                }
            }
            // SPIR-V <= 1.3 OpEntryPoint interface must only contain Input/Output
            // variables; kernel params are bound by argument index instead.
            // Builtin special registers are Input variables, so append them too.
            let mut interface = global_variables
                .into_iter()
                .filter(|id| {
                    match defined_globals
                        .variables_type_check
                        .get(id)
                        .and_then(|t| t.as_ref())
                    {
                        Some((_, ast::StateSpace::Sreg, _)) => true,
                        Some((_, ast::StateSpace::Const, _)) => true,
                        _ => false,
                    }
            }).collect::<Vec<_>>();
            interface.extend(defined_globals.special_registers.interface());
            builder.entry_point(spirv::ExecutionModel::Kernel, fn_id, name, interface);
            fn_id
        }
        ast::MethodName::Func(name) => name,
    };
    builder.begin_function(
        ret_type,
        Some(fn_id),
        spirv::FunctionControl::NONE,
        func_type,
    )?;
    // TODO: re-enable when Intel float control extension works
    /*
    if let Some(denorm_modes) = denorm_information.get(&func_decl.name) {
        for (size_of, denorm_mode) in denorm_modes {
            builder.decorate(
                fn_id,
                spirv::Decoration::FunctionDenormModeINTEL,
                [
                    dr::Operand::LiteralInt32((*size_of as u32) * 8),
                    dr::Operand::FPDenormMode(*denorm_mode),
                ],
            )
        }
    }
    */
    for (name, typ) in func_decl.effective_input_arguments() {
        let result_type = map.get_or_add(builder, typ);
        builder.function_parameter(Some(name), result_type)?;
    }
    Ok(fn_id)
}

fn emit_capabilities(builder: &mut dr::Builder) {
    builder.capability(spirv::Capability::GenericPointer);
    builder.capability(spirv::Capability::Linkage);
    builder.capability(spirv::Capability::Addresses);
    builder.capability(spirv::Capability::Kernel);
    builder.capability(spirv::Capability::Int8);
    builder.capability(spirv::Capability::Int16);
    builder.capability(spirv::Capability::Int64);
    builder.capability(spirv::Capability::Float16);
    builder.capability(spirv::Capability::Float64);
    builder.capability(spirv::Capability::KernelAttributesINTEL);
    // Warp-level primitives (shfl / vote) are lowered to subgroup operations:
    // the shuffle through the Intel subgroup extension, the vote through the
    // core SPIR-V 1.3 group operations.
    builder.capability(spirv::Capability::SubgroupShuffleINTEL);
    builder.capability(spirv::Capability::GroupNonUniformVote);
    builder.capability(spirv::Capability::GroupNonUniformBallot);
    // A kernel that needs the whole 32 lane warp declares
    // `OpExecutionMode SubgroupSize 32`, and that execution mode formally requires
    // this capability. It is declared unconditionally because a module may hold
    // several kernels and the capability is a module-level declaration; it has no
    // effect on a module whose kernels do not use the mode.
    builder.capability(spirv::Capability::SubgroupDispatch);
    // The bitwise warp reductions (`redux.sync.or/and/xor`) lower to the group
    // non-uniform bitwise operations, which are declared under this capability.
    builder.capability(spirv::Capability::GroupNonUniformArithmetic);
    // bfe / bfi are lowered to the bit field operations, which formally need the
    // BitInstructions capability. The bundled SPIR-V headers predate it and a
    // capability cannot be named by number (the builder rejects the value), so a
    // module that uses one omits the declaration. IGC and libigc accept such a
    // module and the kernels execute, so this is a validator note rather than a
    // missing feature; declaring Shader to satisfy the validator instead would
    // make it require structured control flow, which these kernels do not have.
    // TODO: re-enable when Intel float control extension works
    //builder.capability(spirv::Capability::FunctionFloatControlINTEL);
}

// http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/KHR/SPV_KHR_float_controls.html
fn emit_extensions(builder: &mut dr::Builder) {
    builder.extension("SPV_INTEL_kernel_attributes");
    builder.extension("SPV_INTEL_subgroups");
    // TODO: re-enable when Intel float control extension works
    //builder.extension("SPV_INTEL_float_controls2");
}

fn emit_opencl_import(builder: &mut dr::Builder) -> spirv::Word {
    builder.ext_inst_import("OpenCL.std")
}

fn emit_memory_model(builder: &mut dr::Builder) {
    builder.memory_model(
        spirv::AddressingModel::Physical64,
        spirv::MemoryModel::OpenCL,
    );
}

fn translate_directive<'input>(
    id_defs: &mut GlobalStringIdResolver<'input>,
    ptx_impl_imports: &mut HashMap<String, Directive<'input>>,
    d: ast::Directive<'input, ast::ParsedArgParams<'input>>,
) -> Result<Option<Directive<'input>>, TranslateError> {
    Ok(match d {
        ast::Directive::Variable(linking, var) => Some(Directive::Variable(
            linking,
            ast::Variable {
                align: var.align,
                v_type: var.v_type.clone(),
                state_space: var.state_space,
                name: id_defs.get_or_add_def_typed(var.name, var.v_type, var.state_space, true),
                array_init: var.array_init,
            },
        )),
        ast::Directive::Method(_, f) => {
            translate_function(id_defs, ptx_impl_imports, f)?.map(Directive::Method)
        }
    })
}

fn translate_function<'a>(
    id_defs: &mut GlobalStringIdResolver<'a>,
    ptx_impl_imports: &mut HashMap<String, Directive<'a>>,
    f: ast::ParsedFunction<'a>,
) -> Result<Option<Function<'a>>, TranslateError> {
    // __assertfail is the one function that is imported from the implementation
    // module, which defines it with CUDA's own signature: message, file, line,
    // function, character count. A declaration that disagrees is not imported,
    // because the backend's linker aborts on a signature mismatch instead of
    // reporting it; the call gets a stub body like any other function that is
    // called and not defined here, which is what a lift produces for it.
    const ASSERTFAIL_ARGUMENTS: usize = 5;
    let import_as = match &f.func_directive {
        ast::MethodDeclaration {
            name: ast::MethodName::Func("__assertfail"),
            input_arguments,
            ..
        } if input_arguments.len() == ASSERTFAIL_ARGUMENTS => {
            Some("__zluda_ptx_impl____assertfail".to_owned())
        }
        _ => None,
    };
    let (str_resolver, fn_resolver, fn_decl) = id_defs.start_fn(&f.func_directive)?;
    let mut func = to_ssa(
        ptx_impl_imports,
        str_resolver,
        fn_resolver,
        fn_decl,
        f.body,
        f.tuning,
    )?;
    func.import_as = import_as;
    if func.import_as.is_some() {
        ptx_impl_imports.insert(
            func.import_as.as_ref().unwrap().clone(),
            Directive::Method(func),
        );
        Ok(None)
    } else {
        Ok(Some(func))
    }
}

fn rename_fn_params<'a, 'b>(
    fn_resolver: &mut FnStringIdResolver<'a, 'b>,
    args: &'b [ast::Variable<&'a str>],
) -> Vec<ast::Variable<spirv::Word>> {
    args.iter()
        .map(|a| ast::Variable {
            name: fn_resolver.add_def(a.name, Some((a.v_type.clone(), a.state_space)), true),
            v_type: a.v_type.clone(),
            state_space: a.state_space,
            align: a.align,
            array_init: a.array_init.clone(),
        })
        .collect()
}

fn to_ssa<'input, 'b>(
    ptx_impl_imports: &mut HashMap<String, Directive>,
    mut id_defs: FnStringIdResolver<'input, 'b>,
    fn_defs: GlobalFnDeclResolver<'input, 'b>,
    func_decl: Rc<RefCell<ast::MethodDeclaration<'input, spirv::Word>>>,
    f_body: Option<Vec<ast::Statement<ast::ParsedArgParams<'input>>>>,
    tuning: Vec<ast::TuningDirective>,
) -> Result<Function<'input>, TranslateError> {
    let f_body = match f_body {
        Some(vec) => vec,
        None => {
            return Ok(Function {
                func_decl: func_decl,
                body: None,
                globals: Vec::new(),
                import_as: None,
                tuning,
            })
        }
    };
    let normalized_ids = match normalize_identifiers(&mut id_defs, &fn_defs, f_body) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mut numeric_id_defs = id_defs.finish();
    let unadorned_statements = match normalize_predicates(normalized_ids, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let typed_statements = match convert_to_typed_statements(unadorned_statements, &fn_defs, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let (func_decl, typed_statements) = match convert_to_stateful_memory_access(func_decl, typed_statements, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let ssa_statements = match insert_mem_ssa_statements(
        typed_statements,
        &mut numeric_id_defs,
        &mut (*func_decl).borrow_mut(),
    ) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let ssa_statements = match fix_special_registers(ssa_statements, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mut numeric_id_defs = numeric_id_defs.finish();
    let expanded_statements = match expand_arguments(ssa_statements, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let expanded_statements = match insert_implicit_conversions(expanded_statements, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let mut numeric_id_defs = numeric_id_defs.unmut();
    let labeled_statements = normalize_labels(expanded_statements, &mut numeric_id_defs);
    let (f_body, globals) = match extract_globals(labeled_statements, ptx_impl_imports, &mut numeric_id_defs) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok(Function {
        func_decl: func_decl,
        globals: globals,
        body: Some(f_body),
        import_as: None,
        tuning,
    })
}

/// Lower a `%lanemask_*` read into the shifts of the lane id that produce
/// the mask. The mask covers the 32 lanes of a warp:
///   eq: 1 << lane              lt: (1 << lane) - 1
///   le: lt | eq                ge: ~lt
///   gt: ~le
///
/// The one shift is by the lane id, which is always below the width. Writing
/// le as "1 << (lane + 1)" instead was a bug: at lane 31 that is a shift by 32,
/// which PTX defines as 0 but SPIR-V leaves undefined, and the backend resolved
/// it as a shift by 0 - so the mask came out 0 instead of all ones and every
/// lane-31 computation that used it was wrong. Measured on the B580 against an
/// RTX 2070; `tmp/harness/topkdiag/` has the probe.
fn expand_lane_mask(
    numeric_id_defs: &mut NumericIdResolver,
    mask: PtxSpecialRegister,
    dst: spirv::Word,
    typ: ast::ScalarType,
    out: &mut Vec<TypedStatement>,
) -> Result<(), TranslateError> {
    let lane_var = numeric_id_defs
        .special_registers
        .get_or_add(numeric_id_defs.current_id, PtxSpecialRegister::Laneid);
    let lane = numeric_id_defs
        .register_intermediate(Some((ast::Type::Scalar(ast::ScalarType::U32), ast::StateSpace::Reg)));
    out.push(Statement::LoadVar(LoadVarDetails {
        arg: Arg2 {
            dst: lane,
            src: lane_var,
        },
        state_space: ast::StateSpace::Sreg,
        typ: ast::Type::Scalar(ast::ScalarType::U32),
        member_index: None,
    }));
    let one = numeric_id_defs
        .register_intermediate(Some((ast::Type::Scalar(ast::ScalarType::U32), ast::StateSpace::Reg)));
    out.push(Statement::Constant(ConstantDefinition {
        dst: one,
        typ: ast::ScalarType::U32,
        value: ast::ImmediateValue::U64(1),
    }));
    // eq = 1 << lane, the only shift any of the masks needs.
    let eq = numeric_id_defs
        .register_intermediate(Some((ast::Type::Scalar(ast::ScalarType::U32), ast::StateSpace::Reg)));
    out.push(Statement::Instruction(ast::Instruction::Shl(
        ast::ScalarType::U32,
        ast::Arg3 {
            dst: TypedOperand::Reg(eq),
            src1: TypedOperand::Reg(one),
            src2: TypedOperand::Reg(lane),
        },
    )));
    fn u32_reg() -> Option<(ast::Type, ast::StateSpace)> {
        Some((
            ast::Type::Scalar(ast::ScalarType::U32),
            ast::StateSpace::Reg,
        ))
    }
    let value = match mask {
        PtxSpecialRegister::LanemaskEq => eq,
        PtxSpecialRegister::LanemaskLt | PtxSpecialRegister::LanemaskLe | PtxSpecialRegister::LanemaskGe
        | PtxSpecialRegister::LanemaskGt => {
            // lt = eq - 1
            let lt = numeric_id_defs.register_intermediate(u32_reg());
            out.push(Statement::Instruction(ast::Instruction::Sub(
                ast::ArithDetails::Unsigned(ast::ScalarType::U32),
                ast::Arg3 {
                    dst: TypedOperand::Reg(lt),
                    src1: TypedOperand::Reg(eq),
                    src2: TypedOperand::Imm(ast::ImmediateValue::U64(1)),
                },
            )));
            match mask {
                PtxSpecialRegister::LanemaskLt => lt,
                PtxSpecialRegister::LanemaskGe => {
                    // ge = ~lt
                    let ge = numeric_id_defs.register_intermediate(u32_reg());
                    out.push(Statement::Instruction(ast::Instruction::Not(
                        ast::ScalarType::U32,
                        ast::Arg2 {
                            dst: TypedOperand::Reg(ge),
                            src: TypedOperand::Reg(lt),
                        },
                    )));
                    ge
                }
                _ => {
                    // le = lt | eq, and gt is its complement.
                    let le = numeric_id_defs.register_intermediate(u32_reg());
                    out.push(Statement::Instruction(ast::Instruction::Or(
                        ast::ScalarType::U32,
                        ast::Arg3 {
                            dst: TypedOperand::Reg(le),
                            src1: TypedOperand::Reg(lt),
                            src2: TypedOperand::Reg(eq),
                        },
                    )));
                    if matches!(mask, PtxSpecialRegister::LanemaskLe) {
                        le
                    } else {
                        let gt = numeric_id_defs.register_intermediate(u32_reg());
                        out.push(Statement::Instruction(ast::Instruction::Not(
                            ast::ScalarType::U32,
                            ast::Arg2 {
                                dst: TypedOperand::Reg(gt),
                                src: TypedOperand::Reg(le),
                            },
                        )));
                        gt
                    }
                }
            }
        }
        _ => unreachable!(),
    };
    out.push(Statement::Conversion(ImplicitConversion {
        src: value,
        dst,
        from_type: ast::Type::Scalar(ast::ScalarType::U32),
        from_space: ast::StateSpace::Reg,
        to_type: ast::Type::Scalar(typ),
        to_space: ast::StateSpace::Reg,
        kind: ConversionKind::Default,
    }));
    Ok(())
}

fn fix_special_registers(
    typed_statements: Vec<TypedStatement>,
    numeric_id_defs: &mut NumericIdResolver,
) -> Result<Vec<TypedStatement>, TranslateError> {
    let mut result = Vec::with_capacity(typed_statements.len());
    for s in typed_statements {
        // %envreg1/%envreg2 (CUDA kernel parameter-buffer base) has no SPIR-V
        // counterpart. In the kernels we run the composed pointer is stored as
        // dead data and never dereferenced, so lower the read to a zero constant.
        if let Statement::Instruction(ast::Instruction::Mov(_, ref arg)) = s {
            let envreg_read = match arg.src {
                TypedOperand::Reg(src_id) => matches!(
                    numeric_id_defs.special_registers.get(src_id),
                    Some(PtxSpecialRegister::Envreg1 | PtxSpecialRegister::Envreg2)
                ),
                _ => false,
            };
            if envreg_read {
                if let TypedOperand::Reg(dst_id) = arg.dst {
                    result.push(Statement::Constant(ConstantDefinition {
                        dst: dst_id,
                        typ: ast::ScalarType::U32,
                        value: ast::ImmediateValue::U64(0),
                    }));
                    continue;
                }
            }
        }
        match s {
            // A bare %envregN read (no .x/.y member access) is wrapped by
            // InsertMemSSAVisitor into a plain LoadVar before this pass runs;
            // lower it to a zero constant just like the Mov form above.
            Statement::LoadVar(ref details)
                if details.member_index.is_none()
                    && matches!(
                        numeric_id_defs.special_registers.get(details.arg.src),
                        Some(PtxSpecialRegister::Envreg1 | PtxSpecialRegister::Envreg2)
                    ) =>
            {
                result.push(Statement::Constant(ConstantDefinition {
                    dst: details.arg.dst,
                    typ: ast::ScalarType::U32,
                    value: ast::ImmediateValue::U64(0),
                }));
            }
            Statement::LoadVar(ref details)
                if details.member_index.is_none()
                    && matches!(
                        numeric_id_defs.special_registers.get(details.arg.src),
                        Some(
                            PtxSpecialRegister::LanemaskLt
                                | PtxSpecialRegister::LanemaskLe
                                | PtxSpecialRegister::LanemaskEq
                                | PtxSpecialRegister::LanemaskGe
                                | PtxSpecialRegister::LanemaskGt
                        )
                    ) =>
            {
                let mask = match numeric_id_defs.special_registers.get(details.arg.src) {
                    Some(mask) => mask,
                    None => unreachable!(),
                };
                let dst_typ = match &details.typ {
                    ast::Type::Scalar(t) => *t,
                    _ => ast::ScalarType::U32,
                };
                expand_lane_mask(numeric_id_defs, mask, details.arg.dst, dst_typ, &mut result)?;
            }
            Statement::LoadVar(
                mut
                details
                @
                LoadVarDetails {
                    member_index: Some((_, Some(_))),
                    ..
                },
            ) => {
                let index = details.member_index.unwrap().0;
                if index == 3 {
                    result.push(Statement::Constant(ConstantDefinition {
                        dst: details.arg.dst,
                        typ: ast::ScalarType::U32,
                        value: ast::ImmediateValue::U64(0),
                    }));
                } else {
                    let sreg_and_type = match numeric_id_defs.special_registers.get(details.arg.src)
                    {
                        Some(reg) => get_sreg_id_scalar_type(numeric_id_defs, reg),
                        None => None,
                    };
                    let (sreg_src, scalar_typ, vector_width) = match sreg_and_type {
                        Some(sreg_and_type) => sreg_and_type,
                        None => {
                            result.push(Statement::LoadVar(details));
                            continue;
                        }
                    };
                    let temp_id = numeric_id_defs
                        .register_intermediate(Some((details.typ.clone(), details.state_space)));
                    let real_dst = details.arg.dst;
                    details.arg.dst = temp_id;
                    result.push(Statement::LoadVar(LoadVarDetails {
                        arg: Arg2 {
                            src: sreg_src,
                            dst: temp_id,
                        },
                        state_space: ast::StateSpace::Sreg,
                        typ: ast::Type::Scalar(scalar_typ),
                        member_index: Some((index, Some(vector_width))),
                    }));
                    result.push(Statement::Conversion(ImplicitConversion {
                        src: temp_id,
                        dst: real_dst,
                        from_type: ast::Type::Scalar(scalar_typ),
                        from_space: ast::StateSpace::Sreg,
                        to_type: ast::Type::Scalar(ast::ScalarType::U32),
                        to_space: ast::StateSpace::Sreg,
                        kind: ConversionKind::Default,
                    }));
                }
            }
            s => result.push(s),
        }
    }
    Ok(result)
}

fn get_sreg_id_scalar_type(
    numeric_id_defs: &mut NumericIdResolver,
    sreg: PtxSpecialRegister,
) -> Option<(spirv::Word, ast::ScalarType, u8)> {
    match sreg.normalized_sreg_and_type() {
        Some((normalized_sreg, typ, vec_width)) => Some((
            numeric_id_defs
                .special_registers
                .get_or_add(numeric_id_defs.current_id, normalized_sreg),
            typ,
            vec_width,
        )),
        None => None,
    }
}

fn extract_globals<'input, 'b>(
    sorted_statements: Vec<ExpandedStatement>,
    ptx_impl_imports: &mut HashMap<String, Directive>,
    id_def: &mut NumericIdResolver,
) -> Result<(Vec<ExpandedStatement>, Vec<ast::Variable<spirv::Word>>), TranslateError> {
    let mut local = Vec::with_capacity(sorted_statements.len());
    let mut global = Vec::new();
    for statement in sorted_statements {
        match statement {
            Statement::Variable(
                var
                @
                ast::Variable {
                    state_space: ast::StateSpace::Shared,
                    ..
                },
            )
            | Statement::Variable(
                var
                @
                ast::Variable {
                    state_space: ast::StateSpace::Global,
                    ..
                },
            ) => global.push(var),
            Statement::Instruction(ast::Instruction::Bfi { typ, arg }) => {
                let fn_name = [ZLUDA_PTX_PREFIX, "bfi_", typ.to_ptx_name()].concat();
                local.push(instruction_to_fn_call(
                    id_def,
                    ptx_impl_imports,
                    ast::Instruction::Bfi { typ, arg },
                    fn_name,
                )?);
            }
            Statement::Instruction(ast::Instruction::Brev { typ, arg }) => {
                let fn_name = [ZLUDA_PTX_PREFIX, "brev_", typ.to_ptx_name()].concat();
                local.push(instruction_to_fn_call(
                    id_def,
                    ptx_impl_imports,
                    ast::Instruction::Brev { typ, arg },
                    fn_name,
                )?);
            }
            Statement::Instruction(ast::Instruction::Atom(
                details
                @
                ast::AtomDetails {
                    inner:
                        ast::AtomInnerDetails::Unsigned {
                            op: ast::AtomUIntOp::Inc,
                            ..
                        },
                    ..
                },
                args,
            )) => {
                let fn_name = [
                    ZLUDA_PTX_PREFIX,
                    "atom_",
                    details.semantics.to_ptx_name(),
                    "_",
                    details.scope.to_ptx_name(),
                    "_",
                    details.space.to_ptx_name(),
                    "_inc",
                ]
                .concat();
                local.push(instruction_to_fn_call(
                    id_def,
                    ptx_impl_imports,
                    ast::Instruction::Atom(details, args),
                    fn_name,
                )?);
            }
            Statement::Instruction(ast::Instruction::Atom(
                details
                @
                ast::AtomDetails {
                    inner:
                        ast::AtomInnerDetails::Unsigned {
                            op: ast::AtomUIntOp::Dec,
                            ..
                        },
                    ..
                },
                args,
            )) => {
                let fn_name = [
                    ZLUDA_PTX_PREFIX,
                    "atom_",
                    details.semantics.to_ptx_name(),
                    "_",
                    details.scope.to_ptx_name(),
                    "_",
                    details.space.to_ptx_name(),
                    "_dec",
                ]
                .concat();
                local.push(instruction_to_fn_call(
                    id_def,
                    ptx_impl_imports,
                    ast::Instruction::Atom(details, args),
                    fn_name,
                )?);
            }
            Statement::Instruction(ast::Instruction::Atom(
                details
                @
                ast::AtomDetails {
                    inner:
                        ast::AtomInnerDetails::Float {
                            op: ast::AtomFloatOp::Add,
                            ..
                        },
                    ..
                },
                args,
            )) => {
                let fn_name = [
                    ZLUDA_PTX_PREFIX,
                    "atom_",
                    details.semantics.to_ptx_name(),
                    "_",
                    details.scope.to_ptx_name(),
                    "_",
                    details.space.to_ptx_name(),
                    "_add_",
                    details.inner.get_type().to_ptx_name(),
                ]
                .concat();
                local.push(instruction_to_fn_call(
                    id_def,
                    ptx_impl_imports,
                    ast::Instruction::Atom(details, args),
                    fn_name,
                )?);
            }
            s => local.push(s),
        }
    }
    Ok((local, global))
}

impl ast::ScalarType {
    fn to_ptx_name(self) -> &'static str {
        match self {
            ast::ScalarType::B8 => "b8",
            ast::ScalarType::B16 => "b16",
            ast::ScalarType::B32 => "b32",
            ast::ScalarType::B64 => "b64",
            ast::ScalarType::U8 => "u8",
            ast::ScalarType::U16 => "u16",
            ast::ScalarType::U32 => "u32",
            ast::ScalarType::U64 => "u64",
            ast::ScalarType::S8 => "s8",
            ast::ScalarType::S16 => "s16",
            ast::ScalarType::S32 => "s32",
            ast::ScalarType::S64 => "s64",
            ast::ScalarType::F16 => "f16",
            ast::ScalarType::F32 => "f32",
            ast::ScalarType::F64 => "f64",
            ast::ScalarType::F16x2 => "f16x2",
            ast::ScalarType::Pred => "pred",
        }
    }
}

impl ast::AtomSemantics {
    fn to_ptx_name(self) -> &'static str {
        match self {
            ast::AtomSemantics::Relaxed => "relaxed",
            ast::AtomSemantics::Acquire => "acquire",
            ast::AtomSemantics::Release => "release",
            ast::AtomSemantics::AcquireRelease => "acq_rel",
        }
    }
}

impl ast::MemScope {
    fn to_ptx_name(self) -> &'static str {
        match self {
            ast::MemScope::Cta => "cta",
            ast::MemScope::Gpu => "gpu",
            ast::MemScope::Sys => "sys",
        }
    }
}

impl ast::StateSpace {
    fn to_ptx_name(self) -> &'static str {
        match self {
            ast::StateSpace::Generic => "generic",
            ast::StateSpace::Global => "global",
            ast::StateSpace::Shared => "shared",
            ast::StateSpace::Reg => "reg",
            ast::StateSpace::Const => "const",
            ast::StateSpace::Local => "local",
            ast::StateSpace::Param => "param",
            ast::StateSpace::Sreg => "sreg",
        }
    }
}

fn normalize_variable_decls(directives: &mut Vec<Directive>) {
    for directive in directives {
        match directive {
            Directive::Method(Function {
                body: Some(func), ..
            }) => {
                func[1..].sort_by_key(|s| match s {
                    Statement::Variable(_) => 0,
                    _ => 1,
                });
            }
            _ => (),
        }
    }
}

fn convert_to_typed_statements(
    func: Vec<UnconditionalStatement>,
    fn_defs: &GlobalFnDeclResolver,
    id_defs: &mut NumericIdResolver,
) -> Result<Vec<TypedStatement>, TranslateError> {
    let mut result = Vec::<TypedStatement>::with_capacity(func.len());
    for s in func {
        match s {
            Statement::Instruction(inst) => match inst {
                ast::Instruction::Call(call) => {
                    let resolver = match fn_defs.get_fn_sig_resolver(call.func) {
                        Ok(r) => r,
                        Err(e) => {
                            return Err(e);
                        }
                    };
                    let resolved_call = match resolver.resolve_in_spirv_repr(call) {
                        Ok(r) => r,
                        Err(e) => {
                            return Err(e);
                        }
                    };
                    let mut visitor = VectorRepackVisitor::new(&mut result, id_defs);
                    let reresolved_call = resolved_call.visit(&mut visitor)?;
                    visitor.func.push(reresolved_call);
                    visitor.func.extend(visitor.post_stmts);
                }
                inst => {
                    let mut visitor = VectorRepackVisitor::new(&mut result, id_defs);
                    let instruction = match inst.map(&mut visitor) {
                        Ok(v) => Statement::Instruction(v),
                        Err(e) => {
                            return Err(e);
                        }
                    };
                    visitor.func.push(instruction);
                    visitor.func.extend(visitor.post_stmts);
                }
            },
            Statement::Label(i) => result.push(Statement::Label(i)),
            Statement::Variable(v) => result.push(Statement::Variable(v)),
            Statement::Conditional(c) => result.push(Statement::Conditional(c)),
            _ => return Err(error_unreachable()),
        }
    }
    Ok(result)
}

struct VectorRepackVisitor<'a, 'b> {
    func: &'b mut Vec<TypedStatement>,
    id_def: &'b mut NumericIdResolver<'a>,
    post_stmts: Option<TypedStatement>,
}

impl<'a, 'b> VectorRepackVisitor<'a, 'b> {
    fn new(func: &'b mut Vec<TypedStatement>, id_def: &'b mut NumericIdResolver<'a>) -> Self {
        VectorRepackVisitor {
            func,
            id_def,
            post_stmts: None,
        }
    }

    fn convert_vector(
        &mut self,
        is_dst: bool,
        non_default_implicit_conversion: Option<
            fn(
                (ast::StateSpace, &ast::Type),
                (ast::StateSpace, &ast::Type),
            ) -> Result<Option<ConversionKind>, TranslateError>,
        >,
        typ: &ast::Type,
        state_space: ast::StateSpace,
        idx: Vec<spirv::Word>,
    ) -> Result<spirv::Word, TranslateError> {
        // A pack operand either names a vector ("mov.u32 foobar, {a,b}") or
        // the pieces of a wider scalar: "mov.b64 {%r0,%r1}, %rd2" splits 64
        // bits into two 32-bit halves, "mov.b32 {low,high}, %r1" splits 32
        // bits into two 16-bit halves (CUDA's inline assembly for the half
        // conversions). The element type comes from the registers themselves.
        let (scalar_t, width, packed_scalar) = match typ {
            ast::Type::Vector(scalar_t, width) => (*scalar_t, *width, None),
            ast::Type::Scalar(packed) => {
                // The element type comes from the registers of the pack; a
                // hole ("_") has no register, so look at the first real one.
                let first = match idx.iter().find(|id| **id != 0) {
                    Some(id) => *id,
                    None => return Err(error_mismatched_type()),
                };
                let elem_type = match self.id_def.get_typed(first)?.0 {
                    ast::Type::Scalar(s) => s,
                    _ => return Err(error_mismatched_type()),
                };
                let elem_bits = elem_type.size_of() as usize * 8;
                let packed_bits = packed.size_of() as usize * 8;
                if elem_bits * idx.len() != packed_bits {
                    return Err(error_mismatched_type());
                }
                (elem_type, idx.len() as u8, Some(*packed))
            }
            _ => return Err(error_mismatched_type()),
        };
        let temp_vec = self
            .id_def
            .register_intermediate(Some((typ.clone(), state_space)));
        let statement = Statement::RepackVector(RepackVectorDetails {
            is_extract: is_dst,
            typ: scalar_t,
            packed: temp_vec,
            unpacked: idx,
            non_default_implicit_conversion,
            pack_width: width,
            packed_scalar,
        });
        if is_dst {
            self.post_stmts = Some(statement);
        } else {
            self.func.push(statement);
        }
        Ok(temp_vec)
    }
}

impl<'a, 'b> ArgumentMapVisitor<NormalizedArgParams, TypedArgParams>
    for VectorRepackVisitor<'a, 'b>
{
    fn id(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        _: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        Ok(desc.op)
    }

    fn operand(
        &mut self,
        desc: ArgumentDescriptor<ast::Operand<spirv::Word>>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<TypedOperand, TranslateError> {
        Ok(match desc.op {
            ast::Operand::Reg(reg) => TypedOperand::Reg(reg),
            ast::Operand::RegOffset(reg, offset) => TypedOperand::RegOffset(reg, offset),
            ast::Operand::Imm(x) => TypedOperand::Imm(x),
            ast::Operand::VecMember(vec, idx) => TypedOperand::VecMember(vec, idx),
            ast::Operand::VecPack(vec) => TypedOperand::Reg(self.convert_vector(
                desc.is_dst,
                desc.non_default_implicit_conversion,
                typ,
                state_space,
                vec,
            )?),
            // A vector literal of constants: "st.global.v2.b32 [addr], {99, 0};"
            ast::Operand::ImmPack(values) => {
                let (scalar_t, width) = match typ {
                    ast::Type::Vector(scalar_t, width) => (*scalar_t, *width),
                    _ => return Err(error_mismatched_type()),
                };
                let mut ids = Vec::with_capacity(values.len());
                for value in values {
                    let id = self
                        .id_def
                        .register_intermediate(Some((ast::Type::Scalar(scalar_t), state_space)));
                    self.func.push(Statement::Constant(ConstantDefinition {
                        dst: id,
                        typ: scalar_t,
                        value,
                    }));
                    ids.push(id);
                }
                let temp_vec = self
                    .id_def
                    .register_intermediate(Some((typ.clone(), state_space)));
                self.func.push(Statement::RepackVector(RepackVectorDetails {
                    is_extract: false,
                    typ: scalar_t,
                    packed: temp_vec,
                    unpacked: ids,
                    non_default_implicit_conversion: None,
                    pack_width: width,
                    packed_scalar: None,
                }));
                TypedOperand::Reg(temp_vec)
            }
            ast::Operand::MixedPack(elements) => {
                let (scalar_t, width) = match typ {
                    ast::Type::Vector(scalar_t, width) => (*scalar_t, *width),
                    _ => return Err(error_mismatched_type()),
                };
                let mut ids = Vec::with_capacity(elements.len());
                for element in elements {
                    match element {
                        ast::PackElement::Reg(id) => ids.push(id),
                        ast::PackElement::Imm(value) => {
                            let id = self.id_def.register_intermediate(Some((
                                ast::Type::Scalar(scalar_t),
                                state_space,
                            )));
                            self.func.push(Statement::Constant(ConstantDefinition {
                                dst: id,
                                typ: scalar_t,
                                value,
                            }));
                            ids.push(id);
                        }
                    }
                }
                let temp_vec = self
                    .id_def
                    .register_intermediate(Some((typ.clone(), state_space)));
                self.func.push(Statement::RepackVector(RepackVectorDetails {
                    is_extract: false,
                    typ: scalar_t,
                    packed: temp_vec,
                    unpacked: ids,
                    non_default_implicit_conversion: None,
                    pack_width: width,
                    packed_scalar: None,
                }));
                TypedOperand::Reg(temp_vec)
            }
        })
    }
}

fn instruction_to_fn_call(
    id_defs: &mut NumericIdResolver,
    ptx_impl_imports: &mut HashMap<String, Directive>,
    inst: ast::Instruction<ExpandedArgParams>,
    fn_name: String,
) -> Result<ExpandedStatement, TranslateError> {
    let mut arguments = Vec::new();
    inst.visit(&mut |desc: ArgumentDescriptor<spirv::Word>,
                     typ: Option<(&ast::Type, ast::StateSpace)>| {
        let (typ, space) = match typ {
            Some((typ, space)) => (typ.clone(), space),
            None => return Err(error_unreachable()),
        };
        arguments.push((desc, typ, space));
        Ok(0)
    })?;
    let return_arguments_count = arguments
        .iter()
        .position(|(desc, _, _)| !desc.is_dst)
        .unwrap_or(0);
    let (return_arguments, input_arguments) = arguments.split_at(return_arguments_count);
    let fn_id = register_external_fn_call(
        id_defs,
        ptx_impl_imports,
        fn_name,
        return_arguments,
        input_arguments,
    )?;
    Ok(Statement::Call(ResolvedCall {
        uniform: false,
        name: fn_id,
        return_arguments: arguments_to_resolved_arguments(return_arguments),
        input_arguments: arguments_to_resolved_arguments(input_arguments),
    }))
}

fn register_external_fn_call(
    id_defs: &mut NumericIdResolver,
    ptx_impl_imports: &mut HashMap<String, Directive>,
    name: String,
    return_arguments: &[(ArgumentDescriptor<spirv::Word>, ast::Type, ast::StateSpace)],
    input_arguments: &[(ArgumentDescriptor<spirv::Word>, ast::Type, ast::StateSpace)],
) -> Result<spirv::Word, TranslateError> {
    match ptx_impl_imports.entry(name) {
        hash_map::Entry::Vacant(entry) => {
            let fn_id = id_defs.register_intermediate(None);
            let return_arguments = fn_arguments_to_variables(id_defs, return_arguments);
            let input_arguments = fn_arguments_to_variables(id_defs, input_arguments);
            let func_decl = ast::MethodDeclaration::<spirv::Word> {
                return_arguments,
                name: ast::MethodName::Func(fn_id),
                input_arguments,
                shared_mem: None,
                shared_mem_prefix: 0,
            };
            let func = Function {
                func_decl: Rc::new(RefCell::new(func_decl)),
                globals: Vec::new(),
                body: None,
                import_as: Some(entry.key().clone()),
                tuning: Vec::new(),
            };
            entry.insert(Directive::Method(func));
            Ok(fn_id)
        }
        hash_map::Entry::Occupied(entry) => match entry.get() {
            Directive::Method(Function { func_decl, .. }) => match (**func_decl).borrow().name {
                ast::MethodName::Func(fn_id) => Ok(fn_id),
                ast::MethodName::Kernel(_) => Err(error_unreachable()),
            },
            _ => Err(error_unreachable()),
        },
    }
}

fn fn_arguments_to_variables(
    id_defs: &mut NumericIdResolver,
    args: &[(ArgumentDescriptor<spirv::Word>, ast::Type, ast::StateSpace)],
) -> Vec<ast::Variable<spirv::Word>> {
    args.iter()
        .map(|(_, typ, space)| ast::Variable {
            align: None,
            v_type: typ.clone(),
            state_space: *space,
            name: id_defs.register_intermediate(None),
            array_init: Vec::new(),
        })
        .collect::<Vec<_>>()
}

fn arguments_to_resolved_arguments(
    args: &[(ArgumentDescriptor<spirv::Word>, ast::Type, ast::StateSpace)],
) -> Vec<(spirv::Word, ast::Type, ast::StateSpace)> {
    args.iter()
        .map(|(desc, typ, space)| (desc.op, typ.clone(), *space))
        .collect::<Vec<_>>()
}

fn normalize_labels(
    func: Vec<ExpandedStatement>,
    id_def: &mut NumericIdResolver,
) -> Vec<ExpandedStatement> {
    let mut labels_in_use = HashSet::new();
    for s in func.iter() {
        match s {
            Statement::Instruction(i) => {
                if let Some(target) = i.jump_target() {
                    labels_in_use.insert(target);
                }
            }
            Statement::Conditional(cond) => {
                labels_in_use.insert(cond.if_true);
                labels_in_use.insert(cond.if_false);
            }
            Statement::Call(..)
            | Statement::Variable(..)
            | Statement::LoadVar(..)
            | Statement::StoreVar(..)
            | Statement::RetValue(..)
            | Statement::Conversion(..)
            | Statement::Constant(..)
            | Statement::Label(..)
            | Statement::PtrAccess { .. }
            | Statement::RepackVector(..) => {}
        }
    }
    iter::once(Statement::Label(id_def.register_intermediate(None)))
        .chain(func.into_iter().filter(|s| match s {
            Statement::Label(i) => labels_in_use.contains(i),
            _ => true,
        }))
        .collect::<Vec<_>>()
}

fn normalize_predicates(
    func: Vec<NormalizedStatement>,
    id_def: &mut NumericIdResolver,
) -> Result<Vec<UnconditionalStatement>, TranslateError> {
    let mut result = Vec::with_capacity(func.len());
    for s in func {
        match s {
            Statement::Label(id) => result.push(Statement::Label(id)),
            Statement::Instruction((pred, inst)) => {
                if let Some(pred) = pred {
                    let if_true = id_def.register_intermediate(None);
                    let if_false = id_def.register_intermediate(None);
                    let folded_bra = match &inst {
                        ast::Instruction::Bra(_, arg) => Some(arg.src),
                        _ => None,
                    };
                    let mut branch = BrachCondition {
                        predicate: pred.label,
                        if_true: folded_bra.unwrap_or(if_true),
                        if_false,
                    };
                    if pred.not {
                        std::mem::swap(&mut branch.if_true, &mut branch.if_false);
                    }
                    result.push(Statement::Conditional(branch));
                    if folded_bra.is_none() {
                        result.push(Statement::Label(if_true));
                        result.push(Statement::Instruction(inst));
                    }
                    result.push(Statement::Label(if_false));
                } else {
                    result.push(Statement::Instruction(inst));
                }
            }
            Statement::Variable(var) => result.push(Statement::Variable(var)),
            // Blocks are flattened when resolving ids
            _ => return Err(error_unreachable()),
        }
    }
    Ok(result)
}

/*
    How do we handle arguments:
    - input .params in kernels
        .param .b64 in_arg
      get turned into this SPIR-V:
        %1 = OpFunctionParameter %ulong
        %2 = OpVariable %_ptr_Function_ulong Function
             OpStore %2 %1
      We do this for two reasons. One, common treatment for argument-declared
      .param variables and .param variables inside function (we assume that
      at SPIR-V level every .param is a pointer in Function storage class)
    - input .params in functions
        .param .b64 in_arg
      get turned into this SPIR-V:
        %1 = OpFunctionParameter %_ptr_Function_ulong
    - input .regs
        .reg .b64 in_arg
      get turned into the same SPIR-V as kernel .params:
        %1 = OpFunctionParameter %ulong
        %2 = OpVariable %_ptr_Function_ulong Function
             OpStore %2 %1
    - output .regs
        .reg .b64 out_arg
      get just a variable declaration:
        %2 = OpVariable %%_ptr_Function_ulong Function
    - output .params don't exist, they have been moved to input positions
      by an earlier pass
    Distinguishing betweem kernel .params and function .params is not the
    cleanest solution. Alternatively, we could "deparamize" all kernel .param
    arguments by turning them into .reg arguments like this:
        .param .b64 arg -> .reg ptr<.b64,.param> arg
    This has the massive downside that this transformation would have to run
    very early and would muddy up already difficult code. It's simpler to just
    have an if here
*/
fn insert_mem_ssa_statements<'a, 'b>(
    func: Vec<TypedStatement>,
    id_def: &mut NumericIdResolver,
    fn_decl: &'a mut ast::MethodDeclaration<'b, spirv::Word>,
) -> Result<Vec<TypedStatement>, TranslateError> {
    let mut result = Vec::with_capacity(func.len());
    for arg in fn_decl.input_arguments.iter_mut() {
        // IGC rejects array-typed by-value kernel arguments. A by-value
        // struct (e.g. dim3) arrives as `.param .b8 name[12]`, i.e.
        // Array(B8, [12]), and is represented as a vector of u32 with the
        // same byte layout.
        if fn_decl.name.is_kernel() {
            if let ast::Type::Array(ast::ScalarType::B8, dims) = &arg.v_type {
                let bytes: u32 = dims.iter().product();
                // A SPIR-V vector holds at most four components, so a larger
                // struct becomes an OpTypeStruct of 32-bit words. All members
                // are 32-bit with no padding, so the byte layout holds and the
                // body can still read bytes through a pointer bitcast.
                if bytes % 4 == 0 && bytes / 4 <= 4 {
                    arg.v_type = ast::Type::Vector(ast::ScalarType::U32, (bytes / 4) as u8);
                } else if bytes % 4 == 0 {
                    arg.v_type = ast::Type::Struct(vec![
                        ast::ScalarType::U32;
                        (bytes / 4) as usize
                    ]);
                } else if bytes <= 4 {
                    arg.v_type = ast::Type::Vector(ast::ScalarType::U8, bytes as u8);
                }
            }
        }
        insert_mem_ssa_argument(id_def, &mut result, arg, fn_decl.name.is_kernel());
    }
    for arg in fn_decl.return_arguments.iter() {
        insert_mem_ssa_argument_reg_return(&mut result, arg);
    }
    for s in func {
        match s {
            Statement::Call(call) => {
                insert_mem_ssa_statement_default(id_def, &mut result, call.cast())?
            }
            Statement::Instruction(inst) => match inst {
                ast::Instruction::Ret(d) => {
                    // TODO: handle multiple output args
                    match &fn_decl.return_arguments[..] {
                        [return_reg] => {
                            let new_id = id_def.register_intermediate(Some((
                                return_reg.v_type.clone(),
                                ast::StateSpace::Reg,
                            )));
                            result.push(Statement::LoadVar(LoadVarDetails {
                                arg: ast::Arg2 {
                                    dst: new_id,
                                    src: return_reg.name,
                                },
                                // TODO: ret with stateful conversion
                                state_space: ast::StateSpace::Reg,
                                typ: return_reg.v_type.clone(),
                                member_index: None,
                            }));
                            result.push(Statement::RetValue(d, new_id));
                        }
                        [] => result.push(Statement::Instruction(ast::Instruction::Ret(d))),
                        _ => unimplemented!(),
                    }
                }
                inst => insert_mem_ssa_statement_default(id_def, &mut result, inst)?,
            },
            Statement::Conditional(mut bra) => {
                let generated_id = id_def.register_intermediate(Some((
                    ast::Type::Scalar(ast::ScalarType::Pred),
                    ast::StateSpace::Reg,
                )));
                result.push(Statement::LoadVar(LoadVarDetails {
                    arg: Arg2 {
                        dst: generated_id,
                        src: bra.predicate,
                    },
                    state_space: ast::StateSpace::Reg,
                    typ: ast::Type::Scalar(ast::ScalarType::Pred),
                    member_index: None,
                }));
                bra.predicate = generated_id;
                result.push(Statement::Conditional(bra));
            }
            Statement::Conversion(conv) => {
                insert_mem_ssa_statement_default(id_def, &mut result, conv)?
            }
            Statement::PtrAccess(ptr_access) => {
                insert_mem_ssa_statement_default(id_def, &mut result, ptr_access)?
            }
            Statement::RepackVector(repack) => {
                insert_mem_ssa_statement_default(id_def, &mut result, repack)?
            }
            s @ Statement::Variable(_) | s @ Statement::Label(_) => result.push(s),
            // A constant needs no memory SSA processing.
            s @ Statement::Constant(_) => result.push(s),
            _ => return Err(error_unreachable()),
        }
    }
    Ok(result)
}

fn insert_mem_ssa_argument(
    id_def: &mut NumericIdResolver,
    func: &mut Vec<TypedStatement>,
    arg: &mut ast::Variable<spirv::Word>,
    is_kernel: bool,
) {
    if !is_kernel && arg.state_space == ast::StateSpace::Param {
        return;
    }
    let new_id = id_def.register_intermediate(Some((arg.v_type.clone(), arg.state_space)));
    func.push(Statement::Variable(ast::Variable {
        align: arg.align,
        v_type: arg.v_type.clone(),
        state_space: ast::StateSpace::Reg,
        name: arg.name,
        array_init: Vec::new(),
    }));
    func.push(Statement::StoreVar(StoreVarDetails {
        arg: ast::Arg2St {
            src1: arg.name,
            src2: new_id,
        },
        typ: arg.v_type.clone(),
        member_index: None,
    }));
    arg.name = new_id;
}

fn insert_mem_ssa_argument_reg_return(
    func: &mut Vec<TypedStatement>,
    arg: &ast::Variable<spirv::Word>,
) {
    func.push(Statement::Variable(ast::Variable {
        align: arg.align,
        v_type: arg.v_type.clone(),
        state_space: arg.state_space,
        name: arg.name,
        array_init: arg.array_init.clone(),
    }));
}

trait Visitable<From: ArgParamsEx, To: ArgParamsEx>: Sized {
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<From, To>,
    ) -> Result<Statement<ast::Instruction<To>, To>, TranslateError>;
}

struct VisitArgumentDescriptor<
    'a,
    Ctor: FnOnce(spirv::Word) -> Statement<ast::Instruction<U>, U>,
    U: ArgParamsEx,
> {
    desc: ArgumentDescriptor<spirv::Word>,
    typ: &'a ast::Type,
    state_space: ast::StateSpace,
    stmt_ctor: Ctor,
}

impl<
        'a,
        Ctor: FnOnce(spirv::Word) -> Statement<ast::Instruction<U>, U>,
        T: ArgParamsEx<Id = spirv::Word>,
        U: ArgParamsEx<Id = spirv::Word>,
    > Visitable<T, U> for VisitArgumentDescriptor<'a, Ctor, U>
{
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<T, U>,
    ) -> Result<Statement<ast::Instruction<U>, U>, TranslateError> {
        Ok((self.stmt_ctor)(
            visitor.id(self.desc, Some((self.typ, self.state_space)))?,
        ))
    }
}

struct InsertMemSSAVisitor<'a, 'input> {
    id_def: &'a mut NumericIdResolver<'input>,
    func: &'a mut Vec<TypedStatement>,
    post_statements: Vec<TypedStatement>,
}

impl<'a, 'input> InsertMemSSAVisitor<'a, 'input> {
    fn symbol(
        &mut self,
        desc: ArgumentDescriptor<(spirv::Word, Option<u8>)>,
        expected: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        let symbol = desc.op.0;
        if expected.is_none() {
            return Ok(symbol);
        };
        let (mut var_type, var_space, is_variable) = self.id_def.get_typed(symbol)?;
        if !var_space.is_compatible(ast::StateSpace::Reg) || !is_variable {
            return Ok(symbol);
        };
        let member_index = match desc.op.1 {
            Some(idx) => {
                let vector_width = match var_type {
                    ast::Type::Vector(scalar_t, width) => {
                        var_type = ast::Type::Scalar(scalar_t);
                        width
                    }
                    _ => return Err(error_mismatched_type()),
                };
                Some((
                    idx,
                    if self.id_def.special_registers.get(symbol).is_some() {
                        Some(vector_width)
                    } else {
                        None
                    },
                ))
            }
            None => None,
        };
        let generated_id = self
            .id_def
            .register_intermediate(Some((var_type.clone(), ast::StateSpace::Reg)));
        if !desc.is_dst {
            self.func.push(Statement::LoadVar(LoadVarDetails {
                arg: Arg2 {
                    dst: generated_id,
                    src: symbol,
                },
                state_space: ast::StateSpace::Reg,
                typ: var_type,
                member_index,
            }));
        } else {
            self.post_statements
                .push(Statement::StoreVar(StoreVarDetails {
                    arg: Arg2St {
                        src1: symbol,
                        src2: generated_id,
                    },
                    typ: var_type,
                    member_index: member_index.map(|(idx, _)| idx),
                }));
        }
        Ok(generated_id)
    }
}

impl<'a, 'input> ArgumentMapVisitor<TypedArgParams, TypedArgParams>
    for InsertMemSSAVisitor<'a, 'input>
{
    fn id(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        typ: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        self.symbol(desc.new_op((desc.op, None)), typ)
    }

    fn operand(
        &mut self,
        desc: ArgumentDescriptor<TypedOperand>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<TypedOperand, TranslateError> {
        Ok(match desc.op {
            TypedOperand::Reg(reg) => {
                TypedOperand::Reg(self.symbol(desc.new_op((reg, None)), Some((typ, state_space)))?)
            }
            TypedOperand::RegOffset(reg, offset) => TypedOperand::RegOffset(
                self.symbol(desc.new_op((reg, None)), Some((typ, state_space)))?,
                offset,
            ),
            op @ TypedOperand::Imm(..) => op,
            TypedOperand::VecMember(symbol, index) => TypedOperand::Reg(
                self.symbol(desc.new_op((symbol, Some(index))), Some((typ, state_space)))?,
            ),
        })
    }
}

fn insert_mem_ssa_statement_default<'a, 'input, S: Visitable<TypedArgParams, TypedArgParams>>(
    id_def: &'a mut NumericIdResolver<'input>,
    func: &'a mut Vec<TypedStatement>,
    stmt: S,
) -> Result<(), TranslateError> {
    let mut visitor = InsertMemSSAVisitor {
        id_def,
        func,
        post_statements: Vec::new(),
    };
    let new_stmt = match stmt.visit(&mut visitor) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    visitor.func.push(new_stmt);
    visitor.func.extend(visitor.post_statements);
    Ok(())
}

            fn expand_arguments<'a, 'b>(
    func: Vec<TypedStatement>,
    id_def: &'b mut MutableNumericIdResolver<'a>,
) -> Result<Vec<ExpandedStatement>, TranslateError> {
    let mut result = Vec::with_capacity(func.len());
    for s in func {
        match s {
            Statement::Call(call) => {
                let mut visitor = FlattenArguments::new(&mut result, id_def);
                let (new_call, post_stmts) = (call.map(&mut visitor)?, visitor.post_stmts);
                result.push(Statement::Call(new_call));
                result.extend(post_stmts);
            }
            Statement::Instruction(inst) => {
                let mut visitor = FlattenArguments::new(&mut result, id_def);
                let new_inst = match inst.map(&mut visitor) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
                let post_stmts = visitor.post_stmts;
                result.push(Statement::Instruction(new_inst));
                result.extend(post_stmts);
            }
            Statement::Variable(ast::Variable {
                align,
                v_type,
                state_space,
                name,
                array_init,
            }) => result.push(Statement::Variable(ast::Variable {
                align,
                v_type,
                state_space,
                name,
                array_init,
            })),
            Statement::PtrAccess(ptr_access) => {
                let mut visitor = FlattenArguments::new(&mut result, id_def);
                let new_inst = match ptr_access.map(&mut visitor) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
                let post_stmts = visitor.post_stmts;
                result.push(Statement::PtrAccess(new_inst));
                result.extend(post_stmts);
            }
            Statement::RepackVector(repack) => {
                let mut visitor = FlattenArguments::new(&mut result, id_def);
                let new_inst = match repack.map(&mut visitor) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                };
                let post_stmts = visitor.post_stmts;
                result.push(Statement::RepackVector(new_inst));
                result.extend(post_stmts);
            }
            Statement::Label(id) => result.push(Statement::Label(id)),
            Statement::Conditional(bra) => result.push(Statement::Conditional(bra)),
            Statement::LoadVar(details) => result.push(Statement::LoadVar(details)),
            Statement::StoreVar(details) => result.push(Statement::StoreVar(details)),
            Statement::RetValue(d, id) => result.push(Statement::RetValue(d, id)),
            Statement::Conversion(conv) => result.push(Statement::Conversion(conv)),
            Statement::Constant(cnst) => result.push(Statement::Constant(cnst)),
        }
    }
    Ok(result)
}

struct FlattenArguments<'a, 'b> {
    func: &'b mut Vec<ExpandedStatement>,
    id_def: &'b mut MutableNumericIdResolver<'a>,
    post_stmts: Vec<ExpandedStatement>,
}

impl<'a, 'b> FlattenArguments<'a, 'b> {
    fn new(
        func: &'b mut Vec<ExpandedStatement>,
        id_def: &'b mut MutableNumericIdResolver<'a>,
    ) -> Self {
        FlattenArguments {
            func,
            id_def,
            post_stmts: Vec::new(),
        }
    }

    fn reg(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        _: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        Ok(desc.op)
    }

    fn reg_offset(
        &mut self,
        desc: ArgumentDescriptor<(spirv::Word, i32)>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<spirv::Word, TranslateError> {
        let (reg, offset) = desc.op;
        if !desc.is_memory_access {
            let (reg_type, reg_space) = self.id_def.get_typed(reg)?;
            if !reg_space.is_compatible(ast::StateSpace::Reg) {
                return Err(error_mismatched_type());
            }
            let reg_scalar_type = match reg_type {
                ast::Type::Scalar(underlying_type) => underlying_type,
                _ => return Err(error_mismatched_type()),
            };
            let id_constant_stmt = self
                .id_def
                .register_intermediate(reg_type.clone(), ast::StateSpace::Reg);
            self.func.push(Statement::Constant(ConstantDefinition {
                dst: id_constant_stmt,
                typ: reg_scalar_type,
                value: ast::ImmediateValue::S64(offset as i64),
            }));
            let arith_details = match reg_scalar_type.kind() {
                ast::ScalarKind::Signed => ast::ArithDetails::Signed(ast::ArithSInt {
                    typ: reg_scalar_type,
                    saturate: false,
                }),
                ast::ScalarKind::Unsigned | ast::ScalarKind::Bit => {
                    ast::ArithDetails::Unsigned(reg_scalar_type)
                }
                _ => return Err(error_unreachable()),
            };
            let id_add_result = self.id_def.register_intermediate(reg_type, state_space);
            self.func.push(Statement::Instruction(ast::Instruction::Add(
                arith_details,
                ast::Arg3 {
                    dst: id_add_result,
                    src1: reg,
                    src2: id_constant_stmt,
                },
            )));
            Ok(id_add_result)
        } else {
            let id_constant_stmt = self.id_def.register_intermediate(
                ast::Type::Scalar(ast::ScalarType::S64),
                ast::StateSpace::Reg,
            );
            self.func.push(Statement::Constant(ConstantDefinition {
                dst: id_constant_stmt,
                typ: ast::ScalarType::S64,
                value: ast::ImmediateValue::S64(offset as i64),
            }));
            let dst = self.id_def.register_intermediate(typ.clone(), state_space);
            self.func.push(Statement::PtrAccess(PtrAccess {
                underlying_type: typ.clone(),
                state_space: state_space,
                dst,
                ptr_src: reg,
                offset_src: id_constant_stmt,
            }));
            Ok(dst)
        }
    }

    fn immediate(
        &mut self,
        desc: ArgumentDescriptor<ast::ImmediateValue>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<spirv::Word, TranslateError> {
        let scalar_t = if let ast::Type::Scalar(scalar) = typ {
            *scalar
        } else {
            todo!()
        };
        let id = self
            .id_def
            .register_intermediate(ast::Type::Scalar(scalar_t), state_space);
        self.func.push(Statement::Constant(ConstantDefinition {
            dst: id,
            typ: scalar_t,
            value: desc.op,
        }));
        Ok(id)
    }
}

impl<'a, 'b> ArgumentMapVisitor<TypedArgParams, ExpandedArgParams> for FlattenArguments<'a, 'b> {
    fn id(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        t: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        self.reg(desc, t)
    }

    fn operand(
        &mut self,
        desc: ArgumentDescriptor<TypedOperand>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<spirv::Word, TranslateError> {
        match desc.op {
            TypedOperand::Reg(r) => self.reg(desc.new_op(r), Some((typ, state_space))),
            TypedOperand::Imm(x) => self.immediate(desc.new_op(x), typ, state_space),
            TypedOperand::RegOffset(reg, offset) => {
                self.reg_offset(desc.new_op((reg, offset)), typ, state_space)
            }
            TypedOperand::VecMember(..) => Err(error_unreachable()),
        }
    }
}

/*
 There are several kinds of implicit conversions in PTX:
 * auto-bitcast: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#type-information-for-instructions-and-operands
 * special ld/st/cvt conversion rules: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-size-exceeding-instruction-type-size
   - ld.param: not documented, but for instruction `ld.param.<type> x, [y]`,
     semantics are to first zext/chop/bitcast `y` as needed and then do
     documented special ld/st/cvt conversion rules for destination operands
   - st.param [x] y (used as function return arguments) same rule as above applies
   - generic/global ld: for instruction `ld x, [y]`, y must be of type
     b64/u64/s64, which is bitcast to a pointer, dereferenced and then
     documented special ld/st/cvt conversion rules are applied to dst
   - generic/global st: for instruction `st [x], y`, x must be of type
     b64/u64/s64, which is bitcast to a pointer
*/
fn insert_implicit_conversions(
    func: Vec<ExpandedStatement>,
    id_def: &mut MutableNumericIdResolver,
) -> Result<Vec<ExpandedStatement>, TranslateError> {
    let mut result = Vec::with_capacity(func.len());
    let mut idx = 0;
    for s in func.into_iter() {
        idx += 1;
        match s {
            Statement::Call(call) => {
                insert_implicit_conversions_impl(&mut result, id_def, call, idx)?;
            }
            Statement::Instruction(inst) => {
                insert_implicit_conversions_impl(&mut result, id_def, inst, idx)?;
            }
            Statement::PtrAccess(access) => {
                insert_implicit_conversions_impl(&mut result, id_def, access, idx)?;
            }
            Statement::RepackVector(repack) => {
                insert_implicit_conversions_impl(&mut result, id_def, repack, idx)?;
            }
            s @ Statement::Conditional(_)
            | s @ Statement::Conversion(_)
            | s @ Statement::Label(_)
            | s @ Statement::Constant(_)
            | s @ Statement::Variable(_)
            | s @ Statement::LoadVar(..)
            | s @ Statement::StoreVar(..)
            | s @ Statement::RetValue(_, _) => result.push(s),
        }
    }
    Ok(result)
}

fn insert_implicit_conversions_impl(
    func: &mut Vec<ExpandedStatement>,
    id_def: &mut MutableNumericIdResolver,
    stmt: impl Visitable<ExpandedArgParams, ExpandedArgParams>,
    idx: usize,
) -> Result<(), TranslateError> {
    let mut post_conv = Vec::new();
    let statement =
        stmt.visit(&mut |desc: ArgumentDescriptor<spirv::Word>,
                         typ: Option<(&ast::Type, ast::StateSpace)>| {
            let (instr_type, instruction_space) = match typ {
                None => return Ok(desc.op),
                Some(t) => t,
            };
            let (operand_type, operand_space) = id_def.get_typed(desc.op)?;
            let op_sz = operand_type.size_of();
            let conversion_fn = desc
                .non_default_implicit_conversion
                .unwrap_or(default_implicit_conversion);
            let conv_result = conversion_fn(
                (operand_space, &operand_type),
                (instruction_space, instr_type),
            );
            let conv_kind = match conv_result {
                Ok(v) => v,
                Err(TranslateError::MismatchedTypeIn(msg)) => {
                    return Err(TranslateError::MismatchedTypeIn(format!(
                        "{} (instruction wants {}, operand is {})",
                        msg,
                        describe_type(instr_type),
                        describe_type(&operand_type)
                    )))
                }
                Err(e) => return Err(e),
            };
            match conv_kind {
                Some(conv_kind) => {
                    let conv_output = if desc.is_dst {
                        &mut post_conv
                    } else {
                        &mut *func
                    };
                    let mut from_type = instr_type.clone();
                    let mut from_space = instruction_space;
                    let mut to_type = operand_type;
                    let mut to_space = operand_space;
                    let mut src =
                        id_def.register_intermediate(instr_type.clone(), instruction_space);
                    let mut dst = desc.op;
                    let result = Ok(src);
                    if !desc.is_dst {
                        mem::swap(&mut src, &mut dst);
                        mem::swap(&mut from_type, &mut to_type);
                        mem::swap(&mut from_space, &mut to_space);
                    }
                    conv_output.push(Statement::Conversion(ImplicitConversion {
                        src,
                        dst,
                        from_type,
                        from_space,
                        to_type,
                        to_space,
                        kind: conv_kind,
                    }));
                    result
                }
                None => Ok(desc.op),
            }
        })?;
    func.push(statement);
    func.append(&mut post_conv);
    Ok(())
}

/// The width of a PTX warp, in lanes.
const WARP_LANES: u32 = 32;

/// Whether a function exchanges data across the whole warp, which needs a
/// 32-lane subgroup (the default on this GPU is 16).
fn statements_need_full_warp(statements: &[ExpandedStatement]) -> bool {
    statements.iter().any(|statement| match statement {
        Statement::Instruction(ast::Instruction::Shfl(..))
        | Statement::Instruction(ast::Instruction::Vote(..))
        // A ballot is a subgroup operation, so the subgroup has to be the
        // 32 lane warp for the mask to mean what the ISA says.
        | Statement::Instruction(ast::Instruction::Activemask(..))
        | Statement::Instruction(ast::Instruction::LdMatrix(..))
        | Statement::Instruction(ast::Instruction::Mma(..))
        | Statement::Instruction(ast::Instruction::MovMatrix(..))
        | Statement::Instruction(ast::Instruction::Redux(..))
        | Statement::Instruction(ast::Instruction::BarRedPopc(..))
        | Statement::Instruction(ast::Instruction::BarRedPred(..))
        // A warp barrier synchronises the 32 lanes of a PTX warp; with a 16
        // lane subgroup it would only synchronise half of it.
        | Statement::Instruction(ast::Instruction::Bar(ast::BarDetails::WarpSync, _)) => true,
        _ => false,
    })
}

/// Whether the code reads the lane or warp id. Both are per-warp values: with a
/// 16 lane subgroup `%laneid` repeats 0..15 twice and `%warpid` counts halves.
fn special_registers_need_full_warp(special_registers: &SpecialRegistersMap) -> bool {
    special_registers.builtins().any(|(sreg, _)| {
        matches!(
            sreg,
            PtxSpecialRegister::Laneid
                | PtxSpecialRegister::Warpid
                | PtxSpecialRegister::Nwarpid
                | PtxSpecialRegister::LanemaskLt
                | PtxSpecialRegister::LanemaskLe
                | PtxSpecialRegister::LanemaskEq
                | PtxSpecialRegister::LanemaskGe
                | PtxSpecialRegister::LanemaskGt
        )
    })
}

/// Whether the code needs the lane id: the matrix instructions pick the
/// thread that holds a fragment by it and a shuffle computes its source lane.
fn statements_use_laneid(statements: &[ExpandedStatement]) -> bool {
    statements.iter().any(|statement| match statement {
        Statement::Instruction(ast::Instruction::Shfl(..))
        | Statement::Instruction(ast::Instruction::LdMatrix(..))
        | Statement::Instruction(ast::Instruction::Mma(..))
        | Statement::Instruction(ast::Instruction::MovMatrix(..)) => true,
        _ => false,
    })
}

fn statements_use_ldmatrix(statements: &[ExpandedStatement]) -> bool {
    statements.iter().any(|statement| match statement {
        Statement::Instruction(ast::Instruction::LdMatrix(..))
        | Statement::Instruction(ast::Instruction::Mma(..))
        | Statement::Instruction(ast::Instruction::MovMatrix(..)) => true,
        _ => false,
    })
}

fn contains_958(s: &ExpandedStatement) -> bool {
    match s {
        Statement::Instruction(ast::Instruction::St(_, arg)) => arg.src1 == 958 || arg.src2 == 958,
        Statement::Instruction(ast::Instruction::Ld(_, arg)) => arg.dst == 958 || arg.src == 958,
        Statement::Instruction(ast::Instruction::Mov(_, arg)) => arg.dst == 958 || arg.src == 958,
        Statement::Instruction(ast::Instruction::Cvta(_, arg)) => arg.dst == 958 || arg.src == 958,
        Statement::Instruction(ast::Instruction::BarRedPopc(_, a, b)) => *a == 958 || *b == 958,
        Statement::StoreVar(d) => d.arg.src1 == 958 || d.arg.src2 == 958,
        Statement::LoadVar(d) => d.arg.dst == 958 || d.arg.src == 958,
        _ => false,
    }
}

// Reorder the blocks so every one appears after its dominators: rspirv emits
// them in statement order and PTX can put a loop body before its header, which
// spirv-val rejects. Reverse post-order of the CFG satisfies dominance.
fn reorder_blocks_rpo(module: &mut dr::Module, fn_idx: usize) {
    let function = &module.functions[fn_idx];
    if function.blocks.is_empty() {
        return;
    }
    let entry = function.blocks[0].label.as_ref().and_then(|l| l.result_id).unwrap_or(0);
    let mut label_to_block: HashMap<spirv::Word, usize> = HashMap::new();
    for (i, b) in function.blocks.iter().enumerate() {
        if let Some(l) = b.label.as_ref() {
            if let Some(id) = l.result_id {
                label_to_block.insert(id, i);
            }
        }
    }
    let mut order: Vec<usize> = Vec::new();
    let mut visited = HashSet::new();
    fn dfs(
        idx: usize,
        blocks: &[dr::Block],
        label_to_block: &HashMap<spirv::Word, usize>,
        visited: &mut HashSet<usize>,
        order: &mut Vec<usize>,
    ) {
        if !visited.insert(idx) {
            return;
        }
        let block = &blocks[idx];
        let targets = block
            .instructions
            .iter()
            .filter(|i| i.class.opcode == spirv::Op::Branch || i.class.opcode == spirv::Op::BranchConditional || i.class.opcode == spirv::Op::Switch)
            .flat_map(|i| i.operands.iter())
            .filter_map(|o| match o {
                dr::Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .filter_map(|id| label_to_block.get(&id).copied())
            .collect::<Vec<_>>();
        for t in targets {
            dfs(t, blocks, label_to_block, visited, order);
        }
        order.push(idx);
    }
    dfs(
        label_to_block
            .get(&entry)
            .copied()
            .unwrap_or(0),
        &function.blocks,
        &label_to_block,
        &mut visited,
        &mut order,
    );
    // Reverse post-order: entry block first (post-order pushes entry last).
    order.reverse();
    // Also append any blocks not reachable via DFS (e.g. landing pads).
    for i in 0..function.blocks.len() {
        if !visited.contains(&i) {
            order.push(i);
        }
    }
    if order.len() != function.blocks.len() {
        return;
    }
    let old = std::mem::take(&mut module.functions[fn_idx].blocks);
    module.functions[fn_idx].blocks = order.into_iter().map(|i| old[i].clone()).collect();
}

fn get_function_type(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    spirv_input: impl Iterator<Item = SpirvType>,
    spirv_output: &[ast::Variable<spirv::Word>],
) -> (spirv::Word, spirv::Word) {
    map.get_or_add_fn(
        builder,
        spirv_input,
        spirv_output
            .iter()
            .map(|var| SpirvType::new(var.v_type.clone())),
    )
}

fn emit_function_body_ops(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    func: &[ExpandedStatement],
    callee_param_types: &HashMap<spirv::Word, Vec<SpirvType>>,
    laneid: Option<spirv::Word>,
) -> Result<(), TranslateError> {
    for (sidx, s) in func.iter().enumerate() {
        match s {
            Statement::Label(id) => {
                if builder.selected_block().is_some() {
                    builder.branch(*id)?;
                }
                builder.begin_block(Some(*id))?;
            }
            _ => {
                if builder.selected_block().is_none() && builder.selected_function().is_some() {
                    builder.begin_block(None)?;
                }
            }
        }
        match s {
            Statement::Label(_) => (),
            Statement::Call(call) => {
                let (result_type, result_id) = match &*call.return_arguments {
                    [(id, typ, space)] => {
                        if *space != ast::StateSpace::Reg {
                            return Err(error_unreachable());
                        }
                        (
                            map.get_or_add(builder, SpirvType::new(typ.clone())),
                            Some(*id),
                        )
                    }
                    [] => (map.void(), None),
                    _ => todo!(),
                };
                let arg_list = call
                    .input_arguments
                    .iter()
                    .map(|(id, _, _)| *id)
                    .collect::<Vec<_>>();
                builder.function_call(result_type, result_id, call.name, arg_list)?;
            }
            Statement::Variable(var) => {
                emit_variable(builder, map, var)?;
            }
            Statement::Constant(cnst) => {
                let typ_id = map.get_or_add_scalar(builder, cnst.typ);
                match (cnst.typ, cnst.value) {
                    (ast::ScalarType::B8, ast::ImmediateValue::U64(value))
                    | (ast::ScalarType::U8, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u8 as u32);
                    }
                    (ast::ScalarType::B16, ast::ImmediateValue::U64(value))
                    | (ast::ScalarType::U16, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u16 as u32);
                    }
                    (ast::ScalarType::B32, ast::ImmediateValue::U64(value))
                    | (ast::ScalarType::U32, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u32);
                    }
                    (ast::ScalarType::B64, ast::ImmediateValue::U64(value))
                    | (ast::ScalarType::U64, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u64(typ_id, Some(cnst.dst), value);
                    }
                    (ast::ScalarType::S8, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as i8 as i32 as u32);
                    }
                    (ast::ScalarType::S16, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as i16 as i32 as u32);
                    }
                    (ast::ScalarType::S32, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as i32 as u32);
                    }
                    (ast::ScalarType::S64, ast::ImmediateValue::U64(value)) => {
                        builder.constant_u64(typ_id, Some(cnst.dst), value as i64 as u64);
                    }
                    (ast::ScalarType::B8, ast::ImmediateValue::S64(value))
                    | (ast::ScalarType::U8, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u8 as u32);
                    }
                    (ast::ScalarType::B16, ast::ImmediateValue::S64(value))
                    | (ast::ScalarType::U16, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u16 as u32);
                    }
                    // Signed narrow types keep the value truncated to their
                    // width: OpSGreaterThan reinterprets the 16-bit pattern as
                    // signed.
                    (ast::ScalarType::S8, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u8 as u32);
                    }
                    (ast::ScalarType::S16, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u16 as u32);
                    }
                    (ast::ScalarType::B32, ast::ImmediateValue::S64(value))
                    | (ast::ScalarType::U32, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as u32);
                    }
                    (ast::ScalarType::B32, ast::ImmediateValue::F32(value))
                    | (ast::ScalarType::U32, ast::ImmediateValue::F32(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value.to_bits());
                    }
                    // A bit register holding a floating point constant keeps the
                    // bit pattern, the same way the 32-bit form above does.
                    (ast::ScalarType::B64, ast::ImmediateValue::F64(value))
                    | (ast::ScalarType::U64, ast::ImmediateValue::F64(value)) => {
                        builder.constant_u64(typ_id, Some(cnst.dst), value.to_bits());
                    }
                    (ast::ScalarType::B64, ast::ImmediateValue::S64(value))
                    | (ast::ScalarType::U64, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u64(typ_id, Some(cnst.dst), value as u64);
                    }
                    (ast::ScalarType::S8, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as i8 as u32);
                    }
                    (ast::ScalarType::S16, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as i16 as u32);
                    }
                    (ast::ScalarType::S32, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u32(typ_id, Some(cnst.dst), value as i32 as u32);
                    }
                    (ast::ScalarType::S64, ast::ImmediateValue::S64(value)) => {
                        builder.constant_u64(typ_id, Some(cnst.dst), value as u64);
                    }
                    (ast::ScalarType::F16, ast::ImmediateValue::F32(value)) => {
                        builder.constant_f32(typ_id, Some(cnst.dst), f16::from_f32(value).to_f32());
                    }
                    (ast::ScalarType::F32, ast::ImmediateValue::F32(value)) => {
                        builder.constant_f32(typ_id, Some(cnst.dst), value);
                    }
                    (ast::ScalarType::F64, ast::ImmediateValue::F32(value)) => {
                        builder.constant_f64(typ_id, Some(cnst.dst), value as f64);
                    }
                    (ast::ScalarType::F16, ast::ImmediateValue::F64(value)) => {
                        builder.constant_f32(typ_id, Some(cnst.dst), f16::from_f64(value).to_f32());
                    }
                    (ast::ScalarType::F32, ast::ImmediateValue::F64(value)) => {
                        builder.constant_f32(typ_id, Some(cnst.dst), value as f32);
                    }
                    (ast::ScalarType::F64, ast::ImmediateValue::F64(value)) => {
                        builder.constant_f64(typ_id, Some(cnst.dst), value);
                    }
                    (ast::ScalarType::Pred, ast::ImmediateValue::U64(value)) => {
                        let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                        if value == 0 {
                            builder.constant_false(bool_type, Some(cnst.dst));
                        } else {
                            builder.constant_true(bool_type, Some(cnst.dst));
                        }
                    }
                    (ast::ScalarType::Pred, ast::ImmediateValue::S64(value)) => {
                        let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                        if value == 0 {
                            builder.constant_false(bool_type, Some(cnst.dst));
                        } else {
                            builder.constant_true(bool_type, Some(cnst.dst));
                        }
                    }
                    _ => {
                        let kind = match cnst.value {
                            ast::ImmediateValue::U64(_) => "u64",
                            ast::ImmediateValue::S64(_) => "s64",
                            ast::ImmediateValue::F32(_) => "f32",
                            ast::ImmediateValue::F64(_) => "f64",
                        };
                        return Err(TranslateError::MismatchedTypeIn(format!(
                            "constant {} in a {}-byte register",
                            kind,
                            cnst.typ.size_of()
                        )));
                    }
                }
            }
            Statement::Conversion(cv) => {
                emit_implicit_conversion(builder, map, cv)?
            }
            Statement::Conditional(bra) => {
                builder.branch_conditional(
                    bra.predicate,
                    bra.if_true,
                    bra.if_false,
                    iter::empty(),
                )?;
            }
            Statement::Instruction(inst) => match inst {
                ast::Instruction::Abs(d, arg) => emit_abs(builder, map, opencl, d, arg)?,
                ast::Instruction::Call(_) => unreachable!(),
                // A memory barrier, at device scope: see emit_device_fence.
                ast::Instruction::Membar(_) => emit_device_fence(builder, map)?,
                // SPIR-V does not support marking jumps as guaranteed-converged
                ast::Instruction::Bra(_, arg) => {
                    builder.branch(arg.src)?;
                }
                ast::Instruction::LdMatrix(details, arg) => {
                    emit_ldmatrix(builder, map, laneid, details, arg)?
                }
                ast::Instruction::CvtPacked(typ, arg) => {
                    emit_cvt_packed(builder, map, *typ, arg)?
                }
                ast::Instruction::Redux(op, typ, arg) => {
                    emit_redux(builder, map, op, typ, arg)?
                }
                // brx.idx: an indirect branch; an index out of range falls
                // through to the instruction after the branch.
                ast::Instruction::BrxIdx(index, _table, targets) => {
                    let index = *index;
                    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                    let index = builder.bitcast(u32_type, None, index)?;
                    let fallthrough = builder.id();
                    builder.switch(
                        index,
                        fallthrough,
                        targets
                            .iter()
                            .enumerate()
                            .map(|(i, target)| (dr::Operand::LiteralInt32(i as u32), *target)),
                    )?;
                    builder.begin_block(Some(fallthrough))?;
                }
                ast::Instruction::VSub4(arg) => emit_vsub4(builder, map, arg)?,
                ast::Instruction::VSet4(op, arg) => emit_vset4(builder, map, op, arg)?,
                ast::Instruction::MovMatrix(arg) => {
                    emit_movmatrix(builder, map, laneid, arg)?
                }
                ast::Instruction::CpAsync(details, arg) => {
                    emit_cp_async(builder, map, details, arg)?
                }
                ast::Instruction::Prmt(typ, arg) => emit_prmt(builder, map, *typ, arg)?,
                ast::Instruction::Dp4a(typ, arg) => emit_dp4a(builder, map, *typ, arg)?,
                ast::Instruction::Copysign(typ, arg) => {
                    emit_copysign(builder, map, *typ, arg)?
                }
                // A scheduling hint: the thread may yield, which changes
                // nothing about the result.
                ast::Instruction::Nanosleep(_) => {}
                // An abort path that only follows an assert. SPIR-V has no
                // trap, and ending the block here would invalidate the
                // instructions the PTX emits after it.
                ast::Instruction::Trap => {}
                ast::Instruction::Shf(details, arg) => emit_shf(builder, map, details, arg)?,
                ast::Instruction::Mma(details, arg) => {
                    emit_mma(builder, map, laneid, details, arg)?
                }
                // The copies are synchronous, so there is nothing to wait for
                // and no groups to keep track of.
                ast::Instruction::CpAsyncCommitGroup
                | ast::Instruction::CpAsyncWaitGroup(_)
                | ast::Instruction::CpAsyncWaitAll => {}
                ast::Instruction::Ld(data, arg) => {
                    let result_type =
                        map.get_or_add(builder, SpirvType::new(ast::Type::from(data.typ.clone())));
                    // The memory qualifiers map onto SPIR-V's memory access
                    // flags: .volatile is the Volatile flag. The ordering
                    // qualifiers (.acquire, .relaxed) are hints that the
                    // surrounding barrier provides.
                    let mut flags = spirv::MemoryAccess::ALIGNED;
                    match data.qualifier {
                        ast::LdStQualifier::Volatile => flags |= spirv::MemoryAccess::VOLATILE,
                        ast::LdStQualifier::Relaxed(_) | ast::LdStQualifier::Acquire(_) => {}
                        ast::LdStQualifier::Weak => {
                            // .nc needs the NonPrivatePointer flag, which
                            // requires the Vulkan memory model a compute kernel
                            // must not declare, so the load stays plain.
                        }
                    }
                    // A volatile load has to observe what another workgroup
                    // stored, and the Volatile flag alone does not make it: see
                    // emit_device_fence. Shared and local memory cannot be
                    // written by another workgroup, so those keep the plain
                    // load.
                    if data.qualifier == ast::LdStQualifier::Volatile
                        && !matches!(
                            data.state_space,
                            ast::StateSpace::Shared
                                | ast::StateSpace::Local
                                | ast::StateSpace::Const
                        )
                    {
                        emit_device_fence(builder, map)?;
                    }
                    builder.load(
                        result_type,
                        Some(arg.dst),
                        arg.src,
                        Some(flags),
                        [dr::Operand::LiteralInt32(
                            ast::Type::from(data.typ.clone()).size_of() as u32,
                        )]
                        .iter()
                        .cloned(),
                    )?;
                }
                ast::Instruction::St(data, arg) => {
                    let mut flags = spirv::MemoryAccess::ALIGNED;
                    match data.qualifier {
                        ast::LdStQualifier::Volatile => flags |= spirv::MemoryAccess::VOLATILE,
                        ast::LdStQualifier::Relaxed(_) | ast::LdStQualifier::Acquire(_) => {}
                        ast::LdStQualifier::Weak => {}
                    }
                    builder.store(
                        arg.src1,
                        arg.src2,
                        Some(flags),
                        [dr::Operand::LiteralInt32(
                            ast::Type::from(data.typ.clone()).size_of() as u32,
                        )]
                        .iter()
                        .cloned(),
                    )?;
                }
                // SPIR-V does not support ret as guaranteed-converged
                ast::Instruction::Ret(_) => builder.ret()?,
                ast::Instruction::Mov(d, arg) => {
                    let result_type =
                        map.get_or_add(builder, SpirvType::new(ast::Type::from(d.typ.clone())));
                    // copy_object needs matching types and OpBitcast cannot
                    // target a predicate, so only pointer<->integer movs are
                    // bitcast.
                    if d.typ == ast::Type::Scalar(ast::ScalarType::Pred) || !d.src_is_address {
                        builder.copy_object(result_type, Some(arg.dst), arg.src)?;
                    } else {
                        builder.bitcast(result_type, Some(arg.dst), arg.src)?;
                    }
                }
                ast::Instruction::Mul(mul, arg) => match mul {
                    ast::MulDetails::Signed(ref ctr) => {
                        emit_mul_sint(builder, map, opencl, ctr, arg)?
                    }
                    ast::MulDetails::Unsigned(ref ctr) => {
                        emit_mul_uint(builder, map, opencl, ctr, arg)?
                    }
                    ast::MulDetails::Float(ref ctr) => emit_mul_float(builder, map, opencl, ctr, arg)?,
                },
                ast::Instruction::Add(add, arg) => match add {
                    ast::ArithDetails::Signed(ref desc) => {
                        emit_add_int(builder, map, opencl, desc.typ.into(), desc.saturate, arg)?
                    }
                    ast::ArithDetails::Unsigned(ref desc) => {
                        emit_add_int(builder, map, opencl, (*desc).into(), false, arg)?
                    }
                    ast::ArithDetails::Float(desc) => {
                        emit_add_float(builder, map, opencl, desc, arg)?
                    }
                },
                ast::Instruction::Setp(setp, arg) => {
                    emit_setp(builder, map, setp, arg)?;
                    if let Some(dst2) = arg.dst2 {
                        // "p|q" also reports the complement of the comparison.
                        let pred_type = map.get_or_add(
                            builder,
                            SpirvType::Base(SpirvScalarKey::Pred),
                        );
                        builder.logical_not(pred_type, Some(dst2), arg.dst1)?;
                    }
                }
                ast::Instruction::Not(t, a) => {
                    let result_type = map.get_or_add(builder, SpirvType::from(*t));
                    let result_id = Some(a.dst);
                    let operand = a.src;
                    match t {
                        ast::ScalarType::Pred => {
                            logical_not(builder, result_type, result_id, operand)
                        }
                        _ => builder.not(result_type, result_id, operand),
                    }?;
                }
                ast::Instruction::Shl(t, a) => {
                    let full_type = ast::Type::Scalar(*t);
                    let size_of = full_type.size_of();
                    let result_type = map.get_or_add(builder, SpirvType::new(full_type));
                    let offset_src = insert_shift_hack(builder, map, a.src2, size_of)?;
                    builder.shift_left_logical(result_type, Some(a.dst), a.src1, offset_src)?;
                }
                ast::Instruction::Shr(t, a) => {
                    let full_type = ast::ScalarType::from(*t);
                    let size_of = full_type.size_of();
                    let result_type = map.get_or_add_scalar(builder, full_type);
                    let offset_src = insert_shift_hack(builder, map, a.src2, size_of as usize)?;
                    if t.kind() == ast::ScalarKind::Signed {
                        // PTX clamps an over-wide shift to the width, which for
                        // an arithmetic right shift replicates the sign:
                        // clamping the amount to width-1 gives exactly that.
                        let limit = match size_of {
                            8 => map.get_or_add_constant(
                                builder,
                                &ast::Type::Scalar(ast::ScalarType::U64),
                                &vec_repr(size_of as u64 * 8 - 1),
                            )?,
                            2 => map.get_or_add_constant(
                                builder,
                                &ast::Type::Scalar(ast::ScalarType::U16),
                                &vec_repr(size_of as u16 * 8 - 1),
                            )?,
                            _ => map.get_or_add_constant(
                                builder,
                                &ast::Type::Scalar(ast::ScalarType::U32),
                                &vec_repr(size_of as u32 * 8 - 1),
                            )?,
                        };
                        let bool_type =
                            map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                        let below =
                            builder.u_less_than(bool_type, None, offset_src, limit)?;
                        let offset_src =
                            builder.select(result_type, None, below, offset_src, limit)?;
                        builder.shift_right_arithmetic(
                            result_type,
                            Some(a.dst),
                            a.src1,
                            offset_src,
                        )?;
                    } else {
                        builder.shift_right_logical(
                            result_type,
                            Some(a.dst),
                            a.src1,
                            offset_src,
                        )?;
                    }
                }
                ast::Instruction::Cvt(dets, arg) => {
                    emit_cvt(builder, map, opencl, dets, arg)?;
                }
                ast::Instruction::Cvta(_, arg) => {
                    // const/slm/global pointers have the same format as generic
                    // ones, so this is a no-op.
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::B64);
                    builder.copy_object(result_type, Some(arg.dst), arg.src)?;
                }
                // setp.cmp.boolop.type p[|q], a, b, c combines the comparison
                // with the predicate c: p = (a cmp b) op c, and the optional q
                // is the same with c inverted.
                ast::Instruction::SetpBool(details, arg) => {
                    let pred_type =
                        map.get_or_add(builder, SpirvType::Base(SpirvScalarKey::Pred));
                    let cmp = builder.id();
                    let plain = ast::SetpData {
                        typ: details.typ,
                        flush_to_zero: details.flush_to_zero,
                        cmp_op: details.cmp_op,
                    };
                    let cmp_arg = ast::Arg4Setp {
                        dst1: cmp,
                        dst2: None,
                        src1: arg.src1,
                        src2: arg.src2,
                    };
                    emit_setp(builder, map, &plain, &cmp_arg)?;
                    let combine = |builder: &mut dr::Builder,
                                   right: spirv::Word|
                     -> Result<spirv::Word, dr::Error> {
                        match details.bool_op {
                            ast::SetpBoolPostOp::And => {
                                builder.logical_and(pred_type, None, cmp, right)
                            }
                            ast::SetpBoolPostOp::Or => {
                                builder.logical_or(pred_type, None, cmp, right)
                            }
                            ast::SetpBoolPostOp::Xor => {
                                builder.logical_not_equal(pred_type, None, cmp, right)
                            }
                        }
                    };
                    let combined = combine(builder, arg.src3)?;
                    builder.copy_object(pred_type, Some(arg.dst1), combined)?;
                    if let Some(dst2) = arg.dst2 {
                        let inverted = builder.logical_not(pred_type, None, arg.src3)?;
                        let combined = combine(builder, inverted)?;
                        builder.copy_object(pred_type, Some(dst2), combined)?;
                    }
                }
                ast::Instruction::Mad(mad, arg) => match mad {
                    ast::MulDetails::Signed(ref desc) => {
                        emit_mad_sint(builder, map, opencl, desc, arg)?
                    }
                    ast::MulDetails::Unsigned(ref desc) => {
                        emit_mad_uint(builder, map, opencl, desc, arg)?
                    }
                    ast::MulDetails::Float(desc) => {
                        emit_mad_float(builder, map, opencl, desc, arg)?
                    }
                },
                ast::Instruction::Or(t, a) => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::from(*t));
                    if *t == ast::ScalarType::Pred {
                        builder.logical_or(result_type, Some(a.dst), a.src1, a.src2)?;
                    } else {
                        builder.bitwise_or(result_type, Some(a.dst), a.src1, a.src2)?;
                    }
                }
                ast::Instruction::Sub(d, arg) => match d {
                    ast::ArithDetails::Signed(desc) => {
                        emit_sub_int(builder, map, opencl, desc.typ.into(), desc.saturate, arg)?;
                    }
                    ast::ArithDetails::Unsigned(desc) => {
                        emit_sub_int(builder, map, opencl, (*desc).into(), false, arg)?;
                    }
                    ast::ArithDetails::Float(desc) => {
                        emit_sub_float(builder, map, opencl, desc, arg)?;
                    }
                },
                ast::Instruction::Min(d, a) => {
                    emit_min(builder, map, opencl, d, a)?;
                }
                ast::Instruction::Max(d, a) => {
                    emit_max(builder, map, opencl, d, a)?;
                }
                ast::Instruction::Rcp(d, a) => {
                    emit_rcp(builder, map, d, a)?;
                }
                ast::Instruction::And(t, a) => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::from(*t));
                    if *t == ast::ScalarType::Pred {
                        builder.logical_and(result_type, Some(a.dst), a.src1, a.src2)?;
                    } else {
                        builder.bitwise_and(result_type, Some(a.dst), a.src1, a.src2)?;
                    }
                }
                ast::Instruction::Selp(t, a) => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::from(*t));
                    builder.select(result_type, Some(a.dst), a.src3, a.src1, a.src2)?;
                }
                // TODO: implement named barriers
                ast::Instruction::Bar(d, _) => {
                    let scope = match d {
                        // bar.warp.sync synchronizes only the warp, so the
                        // scope must not be Workgroup.
                        ast::BarDetails::WarpSync => spirv::Scope::Subgroup as u32,
                        ast::BarDetails::SyncAligned | ast::BarDetails::RedPopc => {
                            spirv::Scope::Workgroup as u32
                        }
                    };
                    let workgroup_scope = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(scope),
                    )?;
                    let barrier_semantics = match d {
                        ast::BarDetails::SyncAligned | ast::BarDetails::RedPopc | ast::BarDetails::WarpSync => map.get_or_add_constant(
                            builder,
                            &ast::Type::Scalar(ast::ScalarType::U32),
                            &vec_repr(
                                spirv::MemorySemantics::CROSS_WORKGROUP_MEMORY
                                    | spirv::MemorySemantics::WORKGROUP_MEMORY
                                    | spirv::MemorySemantics::SEQUENTIALLY_CONSISTENT,
                            ),
                        )?,
                    };
                    builder.control_barrier(workgroup_scope, workgroup_scope, barrier_semantics)?;
                }
                // bar.red.or.pred / bar.red.and.pred: a barrier plus the
                // disjunction (or conjunction) of a predicate across the
                // workgroup.
                //
                // SPIR-V has no workgroup wide group operation, so the votes
                // are counted in one word of workgroup memory: clear it,
                // barrier, add one vote per lane inside a branch, barrier, read
                // whether the word is non-zero, barrier, and let the voting
                // lanes take their vote back out so a loop can re-run the
                // instruction.
                ast::Instruction::BarRedPred(and, dst, _bar_id, pred) => {
                    let and = *and;
                    let dst = *dst;
                    let pred = *pred;
                    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                    let pred_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                    let zero = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(0u32),
                    )?;
                    // The lanes whose vote is counted: those for which the
                    // predicate holds, or for the conjunction those for which
                    // it does not.
                    let voting = if and {
                        builder.logical_not(pred_type, None, pred)?
                    } else {
                        pred
                    };
                    emit_bar_red(builder, map, voting, |builder, collected| {
                        if and {
                            builder.i_equal(pred_type, Some(dst), collected, zero)?;
                        } else {
                            builder.i_not_equal(pred_type, Some(dst), collected, zero)?;
                        }
                        Ok(())
                    })?;
                    let _ = u32_type;
                }
                // bar.red.popc counts the lanes for which the predicate holds.
                ast::Instruction::BarRedPopc(dst, _bar_id, pred) => {
                    let dst = *dst;
                    let pred = *pred;
                    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                    emit_bar_red(builder, map, pred, |builder, collected| {
                        builder.copy_object(u32_type, Some(dst), collected)?;
                        Ok(())
                    })?;
                }
                // shfl.sync.<mode>.<type> d[|p], a, b, c, membermask
                //
                // PTX's shuffle is a warp-level exchange; the Intel subgroup
                // extensions map onto it directly:
                //   .idx  -> OpSubgroupShuffleINTEL
                //   .up   -> OpSubgroupShuffleUpINTEL
                //   .down -> OpSubgroupShuffleDownINTEL
                //   .bfly -> OpSubgroupShuffleXorINTEL
                // The member mask is ignored: the Intel ops cover the subgroup.
                ast::Instruction::Shfl(mode, typ, arg) => {
                    emit_shfl(builder, map, laneid, *mode, *typ, arg)?
                }
                // activemask.b32 d - the lanes of the warp that are still
                // active. A ballot over a predicate that is true in every lane
                // reports exactly the lanes that reach the instruction, which
                // is what the hardware does (measured: half a warp returned
                // gives 0x0000ffff on both the B580 and an RTX 2070).
                ast::Instruction::Activemask(dst) => {
                    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                    let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                    let subgroup_scope = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(spirv::Scope::Subgroup as u32),
                    )?;
                    let one = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(1u32),
                    )?;
                    let always = builder.i_equal(bool_type, None, one, one)?;
                    // A ballot is a vector of four words covering up to 128
                    // lanes; a warp has 32, so take the first.
                    let ballot_type = map.get_or_add(
                        builder,
                        SpirvType::Vector(SpirvScalarKey::B32, 4),
                    );
                    let ballot = builder.group_non_uniform_ballot(
                        ballot_type,
                        None,
                        subgroup_scope,
                        always,
                    )?;
                    builder.composite_extract(
                        u32_type,
                        Some(*dst),
                        ballot,
                        [0].iter().copied(),
                    )?;
                }
                // vote.sync.mode.pred d, a, membermask   (a is a predicate)
                // vote.sync.ballot.b32 d, a, membermask
                ast::Instruction::Vote(mode, typ, arg) => {
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    let subgroup_scope = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(spirv::Scope::Subgroup as u32),
                    )?;
                    let pred = arg.src;
                    match mode {
                        ast::VoteMode::All => {
                            builder.group_non_uniform_all(
                                result_type,
                                Some(arg.dst),
                                subgroup_scope,
                                pred,
                            )?;
                        }
                        ast::VoteMode::Any => {
                            builder.group_non_uniform_any(
                                result_type,
                                Some(arg.dst),
                                subgroup_scope,
                                pred,
                            )?;
                        }
                        ast::VoteMode::Uni => {
                            // True when the predicate agrees in every active
                            // lane, all-true or all-false.
                            let all = builder.group_non_uniform_all(
                                result_type,
                                None,
                                subgroup_scope,
                                pred,
                            )?;
                            let inverted = builder.logical_not(result_type, None, pred)?;
                            let none = builder.group_non_uniform_all(
                                result_type,
                                None,
                                subgroup_scope,
                                inverted,
                            )?;
                            builder.logical_or(result_type, Some(arg.dst), all, none)?;
                        }
                        ast::VoteMode::Ballot => {
                            // The ballot is a vector of four words covering up
                            // to 128 lanes; a warp has 32, so take the first.
                            let ballot_type = map.get_or_add(
                                builder,
                                SpirvType::Vector(SpirvScalarKey::B32, 4),
                            );
                            let ballot = builder.group_non_uniform_ballot(
                                ballot_type,
                                None,
                                subgroup_scope,
                                pred,
                            )?;
                            let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                            builder.composite_extract(
                                u32_type,
                                Some(arg.dst),
                                ballot,
                                [0].iter().copied(),
                            )?;
                        }
                    }
                }
                ast::Instruction::Atom(details, arg) => {
                    emit_atom(builder, map, details, arg)?;
                }
                // A reduction is an atomic whose result the caller discards, so
                // it is emitted the same way with the result going nowhere.
                ast::Instruction::Red(details, arg) => {
                    let discard = builder.id();
                    emit_atom(
                        builder,
                        map,
                        details,
                        &ast::Arg3 {
                            dst: discard,
                            src1: arg.src1,
                            src2: arg.src2,
                        },
                    )?;
                }
                ast::Instruction::AtomCas(details, arg) => {
                    let result_type = map.get_or_add_scalar(builder, details.typ.into());
                    let memory_const = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(details.scope.to_spirv() as u32),
                    )?;
                    let semantics_const = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U32),
                        &vec_repr(details.semantics.to_spirv().bits()),
                    )?;
                    builder.atomic_compare_exchange(
                        result_type,
                        Some(arg.dst),
                        arg.src1,
                        memory_const,
                        semantics_const,
                        semantics_const,
                        arg.src3,
                        arg.src2,
                    )?;
                }
                ast::Instruction::Div(details, arg) => match details {
                    ast::DivDetails::Unsigned(t) => {
                        let result_type = map.get_or_add_scalar(builder, (*t).into());
                        builder.u_div(result_type, Some(arg.dst), arg.src1, arg.src2)?;
                    }
                    ast::DivDetails::Signed(t) => {
                        let result_type = map.get_or_add_scalar(builder, (*t).into());
                        builder.s_div(result_type, Some(arg.dst), arg.src1, arg.src2)?;
                    }
                    ast::DivDetails::Float(t) => {
                        let result_type = map.get_or_add_scalar(builder, t.typ.into());
                        builder.f_div(result_type, Some(arg.dst), arg.src1, arg.src2)?;
                        emit_float_div_decoration(builder, arg.dst, t.kind);
                    }
                },
                ast::Instruction::Sqrt(details, a) => {
                    emit_sqrt(builder, map, opencl, details, a)?;
                }
                ast::Instruction::Rsqrt(details, a) => {
                    let result_type = map.get_or_add_scalar(builder, details.typ.into());
                    builder.ext_inst(
                        result_type,
                        Some(a.dst),
                        opencl,
                        spirv::CLOp::native_rsqrt as spirv::Word,
                        [dr::Operand::IdRef(a.src)].iter().cloned(),
                    )?;
                }
                ast::Instruction::Neg(details, arg) => {
                    // A packed pair of halves is a vector of two halves, so the
                    // result type comes from the type and both lanes negate.
                    let result_type = map.get_or_add(builder, SpirvType::from(details.typ));
                    let negate_func = if matches!(
                        details.typ.kind(),
                        ast::ScalarKind::Float | ast::ScalarKind::Float2
                    ) {
                        dr::Builder::f_negate
                    } else {
                        dr::Builder::s_negate
                    };
                    negate_func(builder, result_type, Some(arg.dst), arg.src)?;
                }
                ast::Instruction::Sin { arg, .. } => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
                    builder.ext_inst(
                        result_type,
                        Some(arg.dst),
                        opencl,
                        spirv::CLOp::sin as u32,
                        [dr::Operand::IdRef(arg.src)].iter().cloned(),
                    )?;
                }
                ast::Instruction::Cos { arg, .. } => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
                    builder.ext_inst(
                        result_type,
                        Some(arg.dst),
                        opencl,
                        spirv::CLOp::cos as u32,
                        [dr::Operand::IdRef(arg.src)].iter().cloned(),
                    )?;
                }
                ast::Instruction::Lg2 { arg, .. } => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
                    builder.ext_inst(
                        result_type,
                        Some(arg.dst),
                        opencl,
                        spirv::CLOp::log2 as u32,
                        [dr::Operand::IdRef(arg.src)].iter().cloned(),
                    )?;
                }
                ast::Instruction::Ex2 { arg, .. } => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
                    builder.ext_inst(
                        result_type,
                        Some(arg.dst),
                        opencl,
                        spirv::CLOp::exp2 as u32,
                        [dr::Operand::IdRef(arg.src)].iter().cloned(),
                    )?;
                }
                ast::Instruction::Clz { typ, arg } => {
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    builder.ext_inst(
                        result_type,
                        Some(arg.dst),
                        opencl,
                        spirv::CLOp::clz as u32,
                        [dr::Operand::IdRef(arg.src)].iter().cloned(),
                    )?;
                }
                // bfind: index of the most significant set bit. A negative
                // signed input is searched as its complement, so the answer is
                // its most significant zero bit; no such bit reports all ones.
                ast::Instruction::Bfind { typ, arg } => {
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    let pred_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                    let bits: u64 = if typ.size_of() >= 8 { 64 } else { 32 };
                    let zero = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(*typ),
                        &vec_repr(0u32),
                    )?;
                    let source = if typ.kind() == ast::ScalarKind::Signed {
                        let negative = builder.s_less_than(pred_type, None, arg.src, zero)?;
                        let inverted = builder.not(result_type, None, arg.src)?;
                        builder.select(result_type, None, negative, inverted, arg.src)?
                    } else {
                        arg.src
                    };
                    let clz = builder.ext_inst(
                        result_type,
                        None,
                        opencl,
                        spirv::CLOp::clz as u32,
                        [dr::Operand::IdRef(source)].iter().cloned(),
                    )?;
                    // The highest set bit is width - 1 and "no bit set" is
                    // width, so both constants belong to the operand's width.
                    let last = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U64),
                        &vec_repr(bits - 1),
                    )?;
                    let width = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(ast::ScalarType::U64),
                        &vec_repr(bits),
                    )?;
                    let u64_type = map.get_or_add_scalar(builder, ast::ScalarType::U64);
                    // clz is computed in the result type, which is already u64 for
                    // the b64 form; a conversion to the same width is invalid.
                    let clz = if bits == 64 {
                        clz
                    } else {
                        builder.u_convert(u64_type, None, clz)?
                    };
                    let wide_result = if bits == 64 { u64_type } else { result_type };
                    let diff = builder.i_sub(wide_result, None, last, clz)?;
                    let diff = if bits == 64 {
                        diff
                    } else {
                        builder.u_convert(result_type, None, diff)?
                    };
                    let none_set = builder.i_equal(pred_type, None, clz, width)?;
                    let all_ones = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(*typ),
                        &if bits == 64 {
                            vec_repr(u64::MAX).to_vec()
                        } else {
                            vec_repr(u32::MAX).to_vec()
                        },
                    )?;
                    builder.select(
                        result_type,
                        Some(arg.dst),
                        none_set,
                        all_ones,
                        diff,
                    )?;
                }
                ast::Instruction::Brev { typ, arg } => {
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    builder.bit_reverse(result_type, Some(arg.dst), arg.src)?;
                }
                ast::Instruction::Popc { typ, arg } => {
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    builder.bit_count(result_type, Some(arg.dst), arg.src)?;
                }
                ast::Instruction::Xor { typ, arg } => {
                    let builder_fn = match typ {
                        ast::ScalarType::Pred => emit_logical_xor_spirv,
                        _ => dr::Builder::bitwise_xor,
                    };
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    builder_fn(builder, result_type, Some(arg.dst), arg.src1, arg.src2)?;
                }
                // bfe: extract a bit field, sign or zero extended to the
                // register width. The backend only implements the 32-bit
                // OpBitField*Extract (the 64-bit ones call a builtin it does
                // not define), so 64-bit bfe is done with shifts:
                //
                //   shifted = base >> pos               (0 if pos >= width)
                //   masked  = shifted & mask(len)       (len clamped)
                //   signed  = (masked << (width - len)) >> (width - len)
                //
                // Every shift amount is masked to the width first, so none is
                // out of range (which would be undefined).
                ast::Instruction::Bfe { typ, arg } => {
                    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::from(*typ));
                    let signed = typ.kind() == ast::ScalarKind::Signed;
                    if typ.size_of() <= 4 {
                        // Shifts rather than OpBitFieldExtract: that opcode formally
                        // requires the BitInstructions capability, which the bundled
                        // SPIR-V headers predate, and a capability cannot be named by
                        // number (the builder rejects the value). IGC accepts the
                        // undeclared form but spirv-val does not, so the 32-bit case
                        // uses the same shift form as the 64-bit one below. Every
                        // shift amount is clamped below the width first, so no shift
                        // is undefined.
                        let width: u32 = 32;
                        let u_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                        let s_type = map.get_or_add_scalar(builder, ast::ScalarType::S32);
                        let c = |builder: &mut dr::Builder, map: &mut TypeWordMap, v: u32| {
                            map.get_or_add_constant(
                                builder,
                                &ast::Type::Scalar(ast::ScalarType::U32),
                                &vec_repr(v),
                            )
                        };
                        let c31 = c(builder, map, 31)?;
                        let c32 = c(builder, map, 32)?;
                        let c1 = c(builder, map, 1)?;
                        let all_ones = c(builder, map, u32::MAX)?;
                        let zero = c(builder, map, 0)?;
                        let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                        // The field width, clamped to the register width.
                        let len_wide = arg.src3;
                        let over =
                            builder.u_greater_than_equal(bool_type, None, len_wide, c32)?;
                        let len_clamped =
                            builder.select(u_type, None, over, c32, len_wide)?;
                        let len_low = builder.bitwise_and(u_type, None, len_clamped, c31)?;
                        let one_shifted =
                            builder.shift_left_logical(u_type, None, c1, len_low)?;
                        let mask = builder.i_sub(u_type, None, one_shifted, c1)?;
                        let mask = builder.select(u_type, None, over, all_ones, mask)?;
                        // The bit position; a position at or past the width yields 0.
                        let pos_wide = arg.src2;
                        let pos_low = builder.bitwise_and(u_type, None, pos_wide, c31)?;
                        let shifted =
                            builder.shift_right_logical(u_type, None, arg.src1, pos_low)?;
                        let pos_over =
                            builder.u_greater_than_equal(bool_type, None, pos_wide, c32)?;
                        let shifted = builder.select(u_type, None, pos_over, zero, shifted)?;
                        let masked = builder.bitwise_and(u_type, None, shifted, mask)?;
                        if signed {
                            // Sign-extend the field: shift its top bit up, then
                            // arithmetic-shift back.
                            let back = builder.i_sub(u_type, None, c32, len_clamped)?;
                            let top = builder.shift_left_logical(u_type, None, masked, back)?;
                            let top = builder.bitcast(s_type, None, top)?;
                            let extended =
                                builder.shift_right_arithmetic(s_type, None, top, back)?;
                            let extended = builder.bitcast(u_type, None, extended)?;
                            builder.copy_object(result_type, Some(arg.dst), extended)?;
                        } else {
                            builder.copy_object(result_type, Some(arg.dst), masked)?;
                        }
                    } else {
                        let width: u32 = 64;
                        let u64_type = map.get_or_add_scalar(builder, ast::ScalarType::U64);
                        let s64_type = map.get_or_add_scalar(builder, ast::ScalarType::S64);
                        let c = |builder: &mut dr::Builder, map: &mut TypeWordMap, v: u64| {
                            map.get_or_add_constant(
                                builder,
                                &ast::Type::Scalar(ast::ScalarType::U64),
                                &vec_repr(v),
                            )
                        };
                        let c63 = c(builder, map, 63)?;
                        let c64 = c(builder, map, 64)?;
                        let c1 = c(builder, map, 1)?;
                        let all_ones = c(builder, map, u64::MAX)?;
                        let zero = c(builder, map, 0)?;
                        let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                        // The width of the field, clamped to 64. The operands are
                        // already 64-bit here: the implicit conversions widened a
                        // narrower one, and a conversion to its own width is not a
                        // valid SPIR-V instruction.
                        let len_wide = arg.src3;
                        let over = builder.u_greater_than_equal(bool_type, None, len_wide, c64)?;
                        let len_wide = builder.select(u64_type, None, over, c64, len_wide)?;
                        let len_low = builder.bitwise_and(u64_type, None, len_wide, c63)?;
                        let one_shifted = builder.shift_left_logical(u64_type, None, c1, len_low)?;
                        let mask = builder.i_sub(u64_type, None, one_shifted, c1)?;
                        let mask = builder.select(u64_type, None, over, all_ones, mask)?;
                        let pos_wide = arg.src2;
                        let pos_low = builder.bitwise_and(u64_type, None, pos_wide, c63)?;
                        let shifted =
                            builder.shift_right_logical(u64_type, None, arg.src1, pos_low)?;
                        let pos_over =
                            builder.u_greater_than_equal(bool_type, None, pos_wide, c64)?;
                        let shifted = builder.select(u64_type, None, pos_over, zero, shifted)?;
                        let masked = builder.bitwise_and(u64_type, None, shifted, mask)?;
                        if signed {
                            let back = builder.i_sub(u64_type, None, c64, len_wide)?;
                            let top = builder.shift_left_logical(u64_type, None, masked, back)?;
                            let top = builder.bitcast(s64_type, None, top)?;
                            let extended = builder.shift_right_arithmetic(s64_type, None, top, back)?;
                            let extended = builder.bitcast(u64_type, None, extended)?;
                            builder.copy_object(result_type, Some(arg.dst), extended)?;
                        } else {
                            builder.copy_object(result_type, Some(arg.dst), masked)?;
                        }
                    }
                }
                ast::Instruction::Bfi { .. } => {
                    return Err(error_unreachable());
                }
                ast::Instruction::Rem { typ, arg } => {
                    // PTX "rem" is C's remainder (the quotient truncates
                    // towards zero), so the result takes the sign of the
                    // dividend; OpSMod floors instead, which differs for mixed
                    // signs.
                    let builder_fn = if typ.kind() == ast::ScalarKind::Signed {
                        dr::Builder::s_rem
                    } else {
                        dr::Builder::u_mod
                    };
                    let result_type = map.get_or_add_scalar(builder, (*typ).into());
                    builder_fn(builder, result_type, Some(arg.dst), arg.src1, arg.src2)?;
                }
            },
            Statement::LoadVar(details) => {
                emit_load_var(builder, map, details)?;
            }
            Statement::StoreVar(details) => {
                let dst_ptr = match details.member_index {
                    Some(index) => {
                        let result_ptr_type = map.get_or_add(
                            builder,
                            SpirvType::pointer_to(
                                details.typ.clone(),
                                spirv::StorageClass::Function,
                            ),
                        );
                        let index_spirv = map.get_or_add_constant(
                            builder,
                            &ast::Type::Scalar(ast::ScalarType::U32),
                            &vec_repr(index as u32),
                        )?;
                        builder.in_bounds_access_chain(
                            result_ptr_type,
                            None,
                            details.arg.src1,
                            [index_spirv].iter().copied(),
                        )?
                    }
                    None => details.arg.src1,
                };
                builder.store(dst_ptr, details.arg.src2, None, iter::empty())?;
            }
            Statement::RetValue(_, id) => {
                builder.ret_value(*id)?;
            }
            Statement::PtrAccess(PtrAccess {
                underlying_type,
                state_space,
                dst,
                ptr_src,
                offset_src,
            }) => {
                let u8_pointer = map.get_or_add(
                    builder,
                    SpirvType::new(ast::Type::Pointer(ast::ScalarType::U8, *state_space)),
                );
                let result_type = map.get_or_add(
                    builder,
                    SpirvType::pointer_to(underlying_type.clone(), state_space.to_spirv()),
                );
                let ptr_src_u8 = builder.bitcast(u8_pointer, None, *ptr_src)?;
                let temp = builder.in_bounds_ptr_access_chain(
                    u8_pointer,
                    None,
                    ptr_src_u8,
                    *offset_src,
                    iter::empty(),
                )?;
                builder.bitcast(result_type, Some(*dst), temp)?;
            }
            Statement::RepackVector(repack) => {
                if let Some(packed_scalar) = repack.packed_scalar {
                    // The pack names the pieces of a wider scalar; the lowest
                    // piece comes first.
                    let elem_bits = repack.typ.size_of() as u32 * 8;
                    let elem_type = map.get_or_add_scalar(builder, repack.typ);
                    let int_elem_type = map.get_or_add_scalar(
                        builder,
                        match repack.typ.size_of() {
                            2 => ast::ScalarType::U16,
                            _ => ast::ScalarType::U32,
                        },
                    );
                    let int_packed_scalar = match packed_scalar.size_of() {
                        2 => ast::ScalarType::U16,
                        4 => ast::ScalarType::U32,
                        _ => ast::ScalarType::U64,
                    };
                    let int_packed_type = map.get_or_add_scalar(builder, int_packed_scalar);
                    let mask = map.get_or_add_constant(
                        builder,
                        &ast::Type::Scalar(int_packed_scalar),
                        &vec_repr((1u64 << elem_bits) - 1),
                    )?;
                    if repack.is_extract {
                        // The pack is the destination: split the value. It
                        // already has the width the pieces are cut out of, and
                        // SPIR-V has no conversion to a value's own type.
                        // Each piece is cut out of the original by its own
                        // shift: with a running value the second piece read the
                        // same low word as the first (the pieces are emitted
                        // per destination, not per pack).
                        let original =
                            builder.copy_object(int_packed_type, None, repack.packed)?;
                        for (index, dst_id) in repack.unpacked.iter().enumerate() {
                            if *dst_id == 0 {
                                // A hole in the pack: the element is not written.
                                continue;
                            }
                            let shift_bits = elem_bits as usize * index;
                            let source = if shift_bits == 0 || elem_bits >= 64 {
                                original
                            } else {
                                let shift = map.get_or_add_constant(
                                    builder,
                                    &ast::Type::Scalar(int_packed_scalar),
                                    &vec_repr(shift_bits as u64),
                                )?;
                                builder.shift_right_logical(
                                    int_packed_type,
                                    None,
                                    original,
                                    shift,
                                )?
                            };
                            let piece = builder.bitwise_and(int_packed_type, None, source, mask)?;
                            let piece = builder.u_convert(int_elem_type, None, piece)?;
                            if elem_type == int_elem_type {
                                builder.copy_object(elem_type, Some(*dst_id), piece)?;
                            } else {
                                // A half precision piece is an integer in the
                                // register; only its bit pattern matters.
                                builder.bitcast(elem_type, Some(*dst_id), piece)?;
                            }
                        }
                    } else {
                        // The pack is the source: join the pieces. Each piece
                        // goes to the position its index names, so the first
                        // one stays at the bottom of the packed value.
                        let mut joined: Option<spirv::Word> = None;
                        for (index, src_id) in repack.unpacked.iter().enumerate() {
                            let piece = if *src_id == 0 {
                                // A hole in the pack: the piece is undefined.
                                0
                            } else {
                                builder.u_convert(int_packed_type, None, *src_id)?
                            };
                            let piece = builder.bitwise_and(int_packed_type, None, piece, mask)?;
                            let shift_bits = elem_bits as usize * index;
                            let shifted = if shift_bits == 0 || elem_bits >= 64 {
                                piece
                            } else {
                                let shift = map.get_or_add_constant(
                                    builder,
                                    &ast::Type::Scalar(int_packed_scalar),
                                    &vec_repr(shift_bits as u64),
                                )?;
                                builder.shift_left_logical(int_packed_type, None, piece, shift)?
                            };
                            joined = Some(match joined {
                                None => shifted,
                                Some(prev) => {
                                    builder.bitwise_or(int_packed_type, None, prev, shifted)?
                                }
                            });
                        }
                        let joined = joined.unwrap_or(0);
                        let result_type = map.get_or_add_scalar(builder, packed_scalar);
                        let joined = if result_type == int_packed_type {
                            builder.copy_object(result_type, None, joined)?
                        } else {
                            builder.bitcast(result_type, None, joined)?
                        };
                        builder.copy_object(result_type, Some(repack.packed), joined)?;
                    }
                } else if repack.is_extract {
                    let scalar_type = map.get_or_add_scalar(builder, repack.typ);
                    for (index, dst_id) in repack.unpacked.iter().enumerate() {
                        if *dst_id == 0 {
                            // A hole in the pack: the element is not written.
                            continue;
                        }
                        builder.composite_extract(
                            scalar_type,
                            Some(*dst_id),
                            repack.packed,
                            [index as u32].iter().copied(),
                        )?;
                    }
                } else {
                    let vector_type = map.get_or_add(
                        builder,
                        SpirvType::Vector(
                            SpirvScalarKey::from(repack.typ),
                            repack.unpacked.len() as u8,
                        ),
                    );
                    let mut temp_vec = builder.undef(vector_type, None);
                    for (index, src_id) in repack.unpacked.iter().enumerate() {
                        let src_id = if *src_id == 0 {
                            // A hole in the source pack: the element is
                            // undefined, use zero.
                            map.get_or_add_constant(
                                builder,
                                &ast::Type::Scalar(ast::ScalarType::U32),
                                &vec_repr(0u32),
                            )?
                        } else {
                            *src_id
                        };
                        temp_vec = builder.composite_insert(
                            vector_type,
                            None,
                            src_id,
                            temp_vec,
                            [index as u32].iter().copied(),
                        )?;
                    }
                    builder.copy_object(vector_type, Some(repack.packed), temp_vec)?;
                }
            }
        }
    }
    Ok(())
}

// HACK ALERT
// For some reason IGC fails linking if the value and shift size are of different type
//
// Note for `shl` and `shr.u`: PTX defines a count at or past the register width
// as zero, SPIR-V leaves the result of OpShiftLeftLogical undefined there, so a
// backend may mask the count instead - and this one does, differently depending
// on how the instruction is spelled. `%lanemask_le` used to be written as
// (1 << (lane + 1)) - 1 and came out 0 at lane 31 for exactly this reason. No
// lifted kernel in the corpus shifts by a constant that wide, but a register
// count can be, so anything that relies on the zero has to clamp or select for
// it rather than assuming the backend will.
fn insert_shift_hack(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    offset_var: spirv::Word,
    size_of: usize,
) -> Result<spirv::Word, TranslateError> {
    let result_type = match size_of {
        2 => map.get_or_add_scalar(builder, ast::ScalarType::B16),
        8 => map.get_or_add_scalar(builder, ast::ScalarType::B64),
        4 => return Ok(offset_var),
        _ => return Err(error_unreachable()),
    };
    Ok(builder.u_convert(result_type, None, offset_var)?)
}

// TODO: check what kind of assembly do we emit
fn emit_logical_xor_spirv(
    builder: &mut dr::Builder,
    result_type: spirv::Word,
    result_id: Option<spirv::Word>,
    op1: spirv::Word,
    op2: spirv::Word,
) -> Result<spirv::Word, dr::Error> {
    let temp_or = builder.logical_or(result_type, None, op1, op2)?;
    let temp_and = builder.logical_and(result_type, None, op1, op2)?;
    let temp_neg = logical_not(builder, result_type, None, temp_and)?;
    builder.logical_and(result_type, result_id, temp_or, temp_neg)
}

fn emit_copysign(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    typ: ast::ScalarType,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let result_type = map.get_or_add_scalar(builder, typ);
    let bits_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let magnitude_mask = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0x7fff_ffffu32),
    )?;
    let sign_mask = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0x8000_0000u32),
    )?;
    // "Copy sign bit of a into value of b": the magnitude comes from the
    // second operand and the sign from the first, the opposite of OpCopySign.
    let sign_source = builder.bitcast(bits_type, None, arg.src1)?;
    let value_source = builder.bitcast(bits_type, None, arg.src2)?;
    let magnitude = builder.bitwise_and(bits_type, None, value_source, magnitude_mask)?;
    let sign = builder.bitwise_and(bits_type, None, sign_source, sign_mask)?;
    let result = builder.bitwise_or(bits_type, None, magnitude, sign)?;
    builder.bitcast(result_type, Some(arg.dst), result)?;
    Ok(())
}

// dp4a.s32.s32 d, a, b, c;   dp4a.u32.u32 d, a, b, c;
//
// Four byte products summed with c. The signed form sign-extends the bytes,
// the unsigned one zero-extends them.
fn emit_dp4a(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    typ: ast::ScalarType,
    arg: &ast::Arg4<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let result_type = map.get_or_add_scalar(builder, typ);
    let i32_type = map.get_or_add_scalar(builder, ast::ScalarType::S32);
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let signed = typ.kind() == ast::ScalarKind::Signed;
    let mut accumulator = arg.src3;
    for index in 0..4u32 {
        let shift = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(index * 8),
        )?;
        let mask = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(0xffu32),
        )?;
        let mut extract = |builder: &mut dr::Builder, source: spirv::Word| {
            let value = builder.shift_right_logical(u32_type, None, source, shift)?;
            let value = builder.bitwise_and(u32_type, None, value, mask)?;
            if signed {
                // Sign-extend the byte: shift it up and back down.
                let twenty_four = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(24u32),
                )?;
                let value = builder.shift_left_logical(u32_type, None, value, twenty_four)?;
                Ok::<_, TranslateError>(builder.shift_right_arithmetic(
                    i32_type,
                    None,
                    value,
                    twenty_four,
                )?)
            } else {
                Ok(builder.u_convert(i32_type, None, value)?)
            }
        };
        let left = extract(builder, arg.src1)?;
        let right = extract(builder, arg.src2)?;
        let product = builder.i_mul(i32_type, None, left, right)?;
        accumulator = builder.i_add(i32_type, None, accumulator, product)?;
    }
    builder.copy_object(result_type, Some(arg.dst), accumulator)?;
    Ok(())
}

/// Two 16-bit floats packed into one register, the low half first.
fn pack_halves(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    low: spirv::Word,
    high: spirv::Word,
    u32_type: spirv::Word,
    b32_type: spirv::Word,
    f16_type: spirv::Word,
) -> Result<spirv::Word, TranslateError> {
    let u16_type = map.get_or_add_scalar(builder, ast::ScalarType::U16);
    let low_bits = builder.bitcast(u16_type, None, low)?;
    let high_bits = builder.bitcast(u16_type, None, high)?;
    let low_bits = builder.u_convert(u32_type, None, low_bits)?;
    let high_bits = builder.u_convert(u32_type, None, high_bits)?;
    let sixteen = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(16u32),
    )?;
    let high_bits = builder.shift_left_logical(u32_type, None, high_bits, sixteen)?;
    let joined = builder.bitwise_or(u32_type, None, low_bits, high_bits)?;
    Ok(builder.bitcast(b32_type, None, joined)?)
}

/// One half of a packed pair of 16-bit floats: half 0 is the low 16 bits of
/// the register, half 1 the high ones.
fn half_of_pack(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    packed: spirv::Word,
    index: u32,
    u32_type: spirv::Word,
    f32_type: spirv::Word,
    f16_type: spirv::Word,
) -> Result<spirv::Word, TranslateError> {
    let _ = u32_type;
    let f16x2_type = map.get_or_add(builder, SpirvType::Vector(SpirvScalarKey::F16, 2));
    let vector = builder.bitcast(f16x2_type, None, packed)?;
    let half = builder.composite_extract(f16_type, None, vector, [index].iter().copied())?;
    Ok(builder.f_convert(f32_type, None, half)?)
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//     {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
//
// The tensor-core matrix multiply D = A * B + C, with the fragments spread
// over the 32 lanes of the warp. With g = lane/4 and j = lane%4:
//
//   A (m x k, row major), 4 x b32 = 8 halves
//     a0 = A[g][2j..2j+1]     a1 = A[g+8][2j..2j+1]
//     a2 = A[g][2j+8..2j+9]   a3 = A[g+8][2j+8..2j+9]
//   B (k x n, column major), 2 x b32 = 4 halves
//     b0 = B[2j..2j+1][g]     b1 = B[2j+8..2j+9][g]
//   C, D (m x n), 4 x f32
//     c0 = C[g][2j]  c1 = C[g][2j+1]  c2 = C[g+8][2j]  c3 = C[g+8][2j+1]
//
// A thread therefore needs the whole row of A and the whole column of B,
// which live in other lanes: they are fetched with a subgroup shuffle. The
// other element widths follow the same shape with 32 / element_bits elements
// to a register.
//
// shfl.sync.mode.b32  d[|p], a, b, c, membermask;
//
// p says whether the computed source lane was in range. With c[4:0] the
// clamp and c[12:8] the segment mask:
//
//   maxLane = (lane & segmask) | (clamp & ~segmask)  minLane = lane & segmask
//   .up    j = lane - b                  p = j >= maxLane
//   .down  j = lane + b                  p = j <= maxLane
//   .bfly  j = lane ^ b                  p = j <= maxLane
//   .idx   j = minLane | (b & ~segmask)  p = j <= maxLane
//
// An out-of-range thread reads its own register, so the lane index handed to
// the backend's shuffle always exists.
fn emit_shfl(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    laneid: Option<spirv::Word>,
    mode: ast::ShflMode,
    typ: ast::ScalarType,
    arg: &ast::ArgShfl<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let laneid = laneid.ok_or(error_todo())?;
    let result_type = map.get_or_add_scalar(builder, typ.into());
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let pred_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
    let constant = |builder: &mut dr::Builder, map: &mut TypeWordMap, value: u32| {
        map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(value),
        )
    };
    let lane_mask = constant(builder, map, 0x1f)?;
    let segment_shift = constant(builder, map, 8)?;
    let lane = builder.load(u32_type, None, laneid, None, [].iter().cloned())?;
    // b is a five bit lane number or offset; a larger value wraps.
    let offset = builder.bitwise_and(u32_type, None, arg.lane, lane_mask)?;
    let clamp = builder.bitwise_and(u32_type, None, arg.clamp, lane_mask)?;
    let segmask = builder.shift_right_logical(u32_type, None, arg.clamp, segment_shift)?;
    let segmask = builder.bitwise_and(u32_type, None, segmask, lane_mask)?;
    let not_segmask = builder.not(u32_type, None, segmask)?;
    let first_in_segment = builder.bitwise_and(u32_type, None, lane, segmask)?;
    let clamp_high = builder.bitwise_and(u32_type, None, clamp, not_segmask)?;
    let max_lane = builder.bitwise_or(u32_type, None, first_in_segment, clamp_high)?;
    let (source, in_range) = match mode {
        ast::ShflMode::Up => {
            let source = builder.i_sub(u32_type, None, lane, offset)?;
            let above_zero = builder.u_greater_than_equal(pred_type, None, lane, offset)?;
            let below_max = builder.u_greater_than_equal(pred_type, None, source, max_lane)?;
            let in_range =
                builder.logical_and(pred_type, None, above_zero, below_max)?;
            (source, in_range)
        }
        ast::ShflMode::Down => {
            let source = builder.i_add(u32_type, None, lane, offset)?;
            let in_range = builder.u_less_than_equal(pred_type, None, source, max_lane)?;
            (source, in_range)
        }
        ast::ShflMode::Bfly => {
            let source = builder.bitwise_xor(u32_type, None, lane, offset)?;
            let in_range = builder.u_less_than_equal(pred_type, None, source, max_lane)?;
            (source, in_range)
        }
        ast::ShflMode::Idx => {
            let low = builder.bitwise_and(u32_type, None, offset, not_segmask)?;
            let source = builder.bitwise_or(u32_type, None, first_in_segment, low)?;
            let in_range = builder.u_less_than_equal(pred_type, None, source, max_lane)?;
            (source, in_range)
        }
    };
    let source = builder.select(u32_type, None, in_range, source, lane)?;
    builder.subgroup_shuffle_intel(result_type, Some(arg.dst), arg.src, source)?;
    if let Some(pred) = arg.dst_predicate {
        builder.copy_object(pred_type, Some(pred), in_range)?;
    }
    Ok(())
}

fn emit_mma(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    laneid: Option<spirv::Word>,
    details: &ast::MmaDetails,
    arg: &ast::ArgMma<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let laneid = laneid.ok_or(error_todo())?;
    if details.m != 16 || details.n != 8 {
        return Err(error_todo());
    }
    let element_bits: u32 = match details.a {
        ast::ScalarType::F16 | ast::ScalarType::B16 => 16,
        // The tf32 form names its elements .tf32, which the AST keeps as a
        // plain 32-bit value.
        ast::ScalarType::B32 => 32,
        ast::ScalarType::S8 | ast::ScalarType::U8 => 8,
        _ => return Err(error_todo()),
    };
    if details.b != details.a {
        return Err(error_todo());
    }
    let k = details.k as u32;
    let half_accumulator = match (details.dst, details.c) {
        (ast::ScalarType::F32, ast::ScalarType::F32) => false,
        (ast::ScalarType::F16, ast::ScalarType::F16) => true,
        (ast::ScalarType::S32, ast::ScalarType::S32) => false,
        (ast::ScalarType::U32, ast::ScalarType::U32) => false,
        _ => return Err(error_todo()),
    };
    let integer = element_bits == 8;
    // A register holds 32 bits of elements: the four lanes of a row group
    // cover k values 0..(4 * per_register), and every further group of that
    // many k values is one more set of registers.
    let per_register = 32 / element_bits;
    let k_per_set = per_register * 4;
    if k % k_per_set != 0 {
        return Err(error_todo());
    }
    let bfloat16 = details.a == ast::ScalarType::B16;
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let b32_type = map.get_or_add_scalar(builder, ast::ScalarType::B32);
    let f32_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
    let f16_type = map.get_or_add_scalar(builder, ast::ScalarType::F16);
    let f16x2_type = map.get_or_add(builder, SpirvType::Vector(SpirvScalarKey::F16, 2));
    let b32_vector4 = map.get_or_add(builder, SpirvType::Vector(SpirvScalarKey::B32, 4));
    let b32_vector2 = map.get_or_add(builder, SpirvType::Vector(SpirvScalarKey::B32, 2));
    let f32_vector4 = map.get_or_add(builder, SpirvType::Vector(SpirvScalarKey::F32, 4));
    // Only the floating point forms need the OpenCL extended instructions;
    // the integer one multiplies and adds directly.
    let opencl = builder
        .module_ref()
        .ext_inst_imports
        .first()
        .and_then(|i| i.result_id);
    let two = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(2u32),
    )?;
    let three = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(3u32),
    )?;
    let one = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(1u32),
    )?;
    let lane = builder.load(u32_type, None, laneid, None, [].iter().cloned())?;
    // g = lane / 4 (the row group), j = lane % 4 (the column pair).
    let group = builder.shift_right_logical(u32_type, None, lane, two)?;
    let in_group = builder.bitwise_and(u32_type, None, lane, three)?;
    let col0 = builder.shift_left_logical(u32_type, None, in_group, one)?;
    let col1 = builder.i_add(u32_type, None, col0, one)?;
    // Every element of the A fragment is converted to a float up front:
    // fetching a pack and converting its halves afterwards is folded by the
    // backend into half precision arithmetic. A set covers k_per_set k values
    // and two rows; B's sets each cover k_per_set values for one column group.
    let sets = k / k_per_set;
    let a_registers = sets * 2;
    let b_registers = sets;
    let result_registers: u32 = if half_accumulator { 2 } else { 4 };
    // A register carries at most four elements (the 8-bit form).
    let mut a_values = vec![[0u32; 4]; a_registers as usize];
    for register in 0..a_registers {
        let element = fragment_register(builder, b32_type, arg.a, register, a_registers)?;
        for half in 0..per_register {
            a_values[register as usize][half as usize] = fragment_element(
                builder,
                map,
                element,
                half,
                element_bits,
                bfloat16,
                details.a == ast::ScalarType::S8,
                u32_type,
                f32_type,
                f16_type,
                f16x2_type,
            )?;
        }
    }
    let mut b_values = vec![[0u32; 4]; b_registers as usize];
    for register in 0..b_registers {
        let element = fragment_register(builder, b32_type, arg.b, register, b_registers)?;
        for half in 0..per_register {
            b_values[register as usize][half as usize] = fragment_element(
                builder,
                map,
                element,
                half,
                element_bits,
                bfloat16,
                details.a == ast::ScalarType::S8,
                u32_type,
                f32_type,
                f16_type,
                f16x2_type,
            )?;
        }
    }
    let mut outputs = Vec::with_capacity(4);
    for (output_index, (row_high, column)) in [
        (false, col0),
        (false, col1),
        (true, col0),
        (true, col1),
    ]
    .into_iter()
    .enumerate()
    {
        let column: spirv::Word = *column;
        // A half precision accumulator packs two values to a register, a float
        // one has a register per value; both are converted, since the
        // accumulation is done in single precision and rounded back.
        let mut sum = if integer {
            let register = fragment_register(
                builder,
                b32_type,
                arg.c,
                output_index as u32,
                result_registers,
            )?;
            builder.bitcast(u32_type, None, register)?
        } else if half_accumulator {
            let register = builder.composite_extract(
                b32_type,
                None,
                arg.c,
                [output_index as u32 / 2].iter().copied(),
            )?;
            fragment_element(
                builder,
                map,
                register,
                output_index as u32 % 2,
                16,
                false,
                false,
                u32_type,
                f32_type,
                f16_type,
                f16x2_type,
            )?
        } else {
            builder.composite_extract(f32_type, None, arg.c, [output_index as u32].iter().copied())?
        };
        for step in 0u32..k {
            let within = step % k_per_set;
            let element = within % per_register;
            let pair = within / per_register;
            let set = step / k_per_set;
            let a_index = set * 2 + if *row_high { 1 } else { 0 };
            let offset = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(pair),
            )?;
            let group_base = builder.shift_left_logical(u32_type, None, group, two)?;
            let a_lane = builder.i_add(u32_type, None, group_base, offset)?;
            // The integer form exchanges integers; the float forms exchange
            // floats.
            let value_type = if integer { u32_type } else { f32_type };
            let a_value = builder.subgroup_shuffle_intel(
                value_type,
                None,
                a_values[a_index as usize][element as usize],
                a_lane,
            )?;
            let column_base = builder.shift_left_logical(u32_type, None, column, two)?;
            let b_lane = builder.i_add(u32_type, None, column_base, offset)?;
            let b_index = set;
            let b_value = builder.subgroup_shuffle_intel(
                value_type,
                None,
                b_values[b_index as usize][element as usize],
                b_lane,
            )?;
            sum = if integer {
                let product = builder.i_mul(u32_type, None, a_value, b_value)?;
                builder.i_add(u32_type, None, product, sum)?
            } else {
                builder.ext_inst(
                    f32_type,
                    None,
                    opencl.ok_or(error_todo())?,
                    spirv::CLOp::fma as u32,
                    [
                        dr::Operand::IdRef(a_value),
                        dr::Operand::IdRef(b_value),
                        dr::Operand::IdRef(sum),
                    ]
                    .iter()
                    .cloned(),
                )?
            };
        }
        outputs.push(sum);
    }
    if integer {
        let mut registers = Vec::with_capacity(4);
        for value in outputs.iter() {
            registers.push(builder.bitcast(b32_type, None, *value)?);
        }
        builder.composite_construct(b32_vector4, Some(arg.dst), registers.into_iter())?;
    } else if half_accumulator {
        let mut registers = Vec::with_capacity(2);
        for pair in 0..2 {
            let low = builder.f_convert(f16_type, None, outputs[pair * 2])?;
            let high = builder.f_convert(f16_type, None, outputs[pair * 2 + 1])?;
            registers.push(pack_halves(builder, map, low, high, u32_type, b32_type, f16_type)?);
        }
        builder.composite_construct(b32_vector2, Some(arg.dst), registers.into_iter())?;
    } else {
        builder.composite_construct(f32_vector4, Some(arg.dst), outputs.into_iter())?;
    }
    let _ = (one, b32_vector4, f16x2_type);
    Ok(())
}

/// One register of a fragment. A fragment of one register is the register
/// itself: SPIR-V has no vector of one element.
fn fragment_register(
    builder: &mut dr::Builder,
    b32_type: spirv::Word,
    fragment: spirv::Word,
    index: u32,
    count: u32,
) -> Result<spirv::Word, TranslateError> {
    if count == 1 {
        return Ok(fragment);
    }
    Ok(builder.composite_extract(b32_type, None, fragment, [index].iter().copied())?)
}

/// One element of a fragment register, as a float.
fn fragment_element(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    register: spirv::Word,
    element: u32,
    element_bits: u32,
    bfloat16: bool,
    signed: bool,
    u32_type: spirv::Word,
    f32_type: spirv::Word,
    f16_type: spirv::Word,
    f16x2_type: spirv::Word,
) -> Result<spirv::Word, TranslateError> {
    if element_bits == 32 {
        // A tf32 element is a 32-bit float; only the rounding of the multiply
        // differs from the float form, and the accumulator is a float.
        return Ok(builder.bitcast(f32_type, None, register)?);
    }
    if element_bits == 8 {
        // Four bytes are packed into the register; the value is widened to a
        // full 32-bit integer of the instruction's sign.
        let shift = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(element * 8),
        )?;
        let mask = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(0xffu32),
        )?;
        let byte = builder.shift_right_logical(u32_type, None, register, shift)?;
        let byte = builder.bitwise_and(u32_type, None, byte, mask)?;
        let u8_type = map.get_or_add_scalar(builder, ast::ScalarType::U8);
        let byte = builder.u_convert(u8_type, None, byte)?;
        return Ok(if signed {
            let s8_type = map.get_or_add_scalar(builder, ast::ScalarType::S8);
            let value = builder.bitcast(s8_type, None, byte)?;
            builder.s_convert(u32_type, None, value)?
        } else {
            builder.u_convert(u32_type, None, byte)?
        });
    }
    if bfloat16 {
        // A bf16 value is the upper half of an f32, so the conversion is a
        // shift of the bit pattern.
        let bits = if element == 0 {
            let mask = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(0xffffu32),
            )?;
            builder.bitwise_and(u32_type, None, register, mask)?
        } else {
            let sixteen = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(16u32),
            )?;
            builder.shift_right_logical(u32_type, None, register, sixteen)?
        };
        let sixteen = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(16u32),
        )?;
        let bits = builder.shift_left_logical(u32_type, None, bits, sixteen)?;
        return Ok(builder.bitcast(f32_type, None, bits)?);
    }
    let vector = builder.bitcast(f16x2_type, None, register)?;
    let half = builder.composite_extract(f16_type, None, vector, [element].iter().copied())?;
    Ok(builder.f_convert(f32_type, None, half)?)
}

// shf.l.wrap.b32 d, a, b, c;   d = (b << c) | (a >> (32 - c))
// shf.r.wrap.b32 d, a, b, c;   d = (a >> c) | (b << (32 - c))
//
// The two operands form one 64-bit value; the result is its low 32 bits
// shifted by c (a shift by 32 is undefined, hence the select at the end).
fn emit_shf(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    details: &ast::ShfDetails,
    arg: &ast::Arg4<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::B32);
    let thirty_one = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(31u32),
    )?;
    let thirty_two = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(32u32),
    )?;
    let zero = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0u32),
    )?;
    let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
    let count = builder.bitwise_and(u32_type, None, arg.src3, thirty_one)?;
    let lowered_shift = builder.i_sub(u32_type, None, thirty_two, count)?;
    let lowered_shift = builder.bitwise_and(u32_type, None, lowered_shift, thirty_one)?;
    // {b, a} is one 64-bit value with b in the high word, so:
    //   shf.l: d = (b << c) | (a >> (32 - c))
    //   shf.r: d = (a >> c) | (b << (32 - c))
    let combined = if details.left {
        let moved_high = builder.shift_left_logical(u32_type, None, arg.src2, count)?;
        let moved_low = builder.shift_right_logical(u32_type, None, arg.src1, lowered_shift)?;
        builder.bitwise_or(u32_type, None, moved_high, moved_low)?
    } else {
        let moved_low = builder.shift_right_logical(u32_type, None, arg.src1, count)?;
        let moved_high = builder.shift_left_logical(u32_type, None, arg.src2, lowered_shift)?;
        builder.bitwise_or(u32_type, None, moved_low, moved_high)?
    };
    if details.wrap {
        // A wrap of zero leaves the value unshifted, so the result is the low
        // word of the pair: b for a left shift, a for a right one.
        let is_zero = builder.i_equal(bool_type, None, count, zero)?;
        let survivor = if details.left { arg.src2 } else { arg.src1 };
        builder.select(u32_type, Some(arg.dst), is_zero, survivor, combined)?;
    } else {
        // A clamp past the width returns the word that survives the shift; a
        // clamp of zero is the wrap-of-zero case, so the low word is the
        // result.
        let wide = builder.u_greater_than_equal(bool_type, None, arg.src3, thirty_two)?;
        let survivor = if details.left { arg.src1 } else { arg.src2 };
        let clamped = builder.select(u32_type, None, wide, survivor, combined)?;
        let is_zero = builder.i_equal(bool_type, None, arg.src3, zero)?;
        let unshifted = if details.left { arg.src2 } else { arg.src1 };
        builder.select(u32_type, Some(arg.dst), is_zero, unshifted, clamped)?;
    }
    Ok(())
}

// prmt.b32 d, a, b, c;
//
// The eight bytes of a and b are one pool (a holds bytes 0-3, b bytes 4-7);
// each nibble of c selects one, the low nibble for the lowest byte. A nibble
// with its top bit set does not select a byte: it replicates the sign bit of
// the selected byte over the whole output byte.
/// bar.red reduces a predicate (or counts the lanes that hold it) across the
/// whole workgroup: the votes are counted in workgroup memory, the total is
/// handed to `finish`, and the word is cleared again so that a loop can run
/// the same instruction once more.
fn emit_bar_red(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    voting: spirv::Word,
    finish: impl FnOnce(&mut dr::Builder, spirv::Word) -> Result<(), TranslateError>,
) -> Result<(), TranslateError> {
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let scratch = builder.id();
    let scratch_type = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::U32),
            spirv::StorageClass::Workgroup,
        ),
    );
    builder.insert_types_global_values(
        dr::InsertPoint::End,
        dr::Instruction::new(
            spirv::Op::Variable,
            Some(scratch_type),
            Some(scratch),
            vec![dr::Operand::StorageClass(spirv::StorageClass::Workgroup)],
        ),
    );
    let workgroup_scope = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(spirv::Scope::Workgroup as u32),
    )?;
    let barrier_semantics = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(
            spirv::MemorySemantics::CROSS_WORKGROUP_MEMORY
                | spirv::MemorySemantics::WORKGROUP_MEMORY
                | spirv::MemorySemantics::SEQUENTIALLY_CONSISTENT,
        ),
    )?;
    let atomic_scope = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(spirv::Scope::Device as u32),
    )?;
    let atomic_semantics = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(spirv::MemorySemantics::NONE.bits()),
    )?;
    let zero = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0u32),
    )?;
    let one = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(1u32),
    )?;
    let barrier = |builder: &mut dr::Builder| -> Result<(), TranslateError> {
        builder.control_barrier(workgroup_scope, workgroup_scope, barrier_semantics)?;
        Ok(())
    };
    // One guarded atomic, in a block of its own. Deliberately not an
    // OpSelectionMerge: the backend predicates a structured selection, and a
    // predicated atomic only lands for a single lane.
    let guarded =
        |builder: &mut dr::Builder, add: bool| -> Result<(), TranslateError> {
            let then_id = builder.id();
            let merge_id = builder.id();
            builder.branch_conditional(voting, then_id, merge_id, iter::empty())?;
            builder.begin_block(Some(then_id))?;
            if add {
                builder.atomic_i_add(
                    u32_type,
                    None,
                    scratch,
                    atomic_scope,
                    atomic_semantics,
                    one,
                )?;
            } else {
                // Clearing the own vote: and with zero.
                builder.atomic_and(
                    u32_type,
                    None,
                    scratch,
                    atomic_scope,
                    atomic_semantics,
                    zero,
                )?;
            }
            builder.branch(merge_id)?;
            builder.begin_block(Some(merge_id))?;
            Ok(())
        };
    // Workgroup memory starts undefined, so the word is cleared before the first
    // vote (every thread writes the same value, so the race is harmless).
    builder.store(
        scratch,
        zero,
        Some(spirv::MemoryAccess::VOLATILE),
        [].iter().cloned(),
    )?;
    barrier(builder)?;
    guarded(builder, true)?;
    barrier(builder)?;
    let collected = builder.load(
        u32_type,
        None,
        scratch,
        Some(spirv::MemoryAccess::VOLATILE),
        [].iter().cloned(),
    )?;
    finish(builder, collected)?;
    barrier(builder)?;
    guarded(builder, false)?;
    barrier(builder)?;
    Ok(())
}

fn emit_prmt(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    typ: ast::ScalarType,
    arg: &ast::Arg4<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let typ: ast::ScalarType = typ.into();
    let result_type = map.get_or_add_scalar(builder, typ);
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
    let constant = |builder: &mut dr::Builder, map: &mut TypeWordMap, value: u32| {
        map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(value),
        )
    };
    let zero = constant(builder, map, 0)?;
    let one = constant(builder, map, 1)?;
    let three = constant(builder, map, 3)?;
    let four = constant(builder, map, 4)?;
    let seven = constant(builder, map, 7)?;
    let eight = constant(builder, map, 8)?;
    let fifteen = constant(builder, map, 15)?;
    let sign = constant(builder, map, 0x80)?;
    let all_ones = constant(builder, map, 0xff)?;
    let mut result: Option<spirv::Word> = None;
    // Each output byte comes from one of the eight input bytes: the selector's
    // low three bits pick the byte, its fourth bit asks for that byte's sign to
    // be replicated instead.
    for index in 0..4u32 {
        let selector_shift = constant(builder, map, index * 4)?;
        let selector = builder.shift_right_logical(u32_type, None, arg.src3, selector_shift)?;
        let selector = builder.bitwise_and(u32_type, None, selector, fifteen)?;
        let byte_index = builder.bitwise_and(u32_type, None, selector, seven)?;
        // Bytes 0..3 are in the first operand, 4..7 in the second.
        let from_second = builder.bitwise_and(u32_type, None, byte_index, four)?;
        let from_second = builder.i_not_equal(bool_type, None, from_second, zero)?;
        let source = builder.select(u32_type, None, from_second, arg.src2, arg.src1)?;
        let shift = builder.bitwise_and(u32_type, None, byte_index, three)?;
        let shift = builder.shift_left_logical(u32_type, None, shift, three)?;
        let byte = builder.shift_right_logical(u32_type, None, source, shift)?;
        let byte = builder.bitwise_and(u32_type, None, byte, all_ones)?;
        // The selector's fourth bit: replicate the byte's sign instead.
        let replicate = builder.bitwise_and(u32_type, None, selector, eight)?;
        let replicate = builder.i_not_equal(bool_type, None, replicate, zero)?;
        let negative = builder.bitwise_and(u32_type, None, byte, sign)?;
        let negative = builder.i_not_equal(bool_type, None, negative, zero)?;
        let ones = builder.select(u32_type, None, negative, all_ones, zero)?;
        let byte = builder.select(u32_type, None, replicate, ones, byte)?;
        result = Some(match result {
            None => byte,
            Some(previous) => {
                let into_place = constant(builder, map, index * 8)?;
                let byte = builder.shift_left_logical(u32_type, None, byte, into_place)?;
                builder.bitwise_or(u32_type, None, previous, byte)?
            }
        });
    }
    let _ = one;
    let result = result.ok_or(error_todo())?;
    builder.copy_object(result_type, Some(arg.dst), result)?;
    Ok(())
}

// cp.async.cg.shared.global [dst], [src], 16;
//
// A plain copy produces the same result (the point of the instruction is
// overlapping it with the arithmetic). The destination is a 32-bit offset
// into the shared window, which is what a Workgroup pointer means in SPIR-V.
fn emit_cp_async(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    details: &ast::CpAsyncDetails,
    arg: &ast::ArgCpAsync<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let size = details.size as u32;
    if size == 0 || size % 4 != 0 || size > 16 {
        return Err(error_todo());
    }
    // The four operand form carries a source size: that many bytes are copied
    // and the rest is zeroed, one 32-bit word at a time below.
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let u8_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B8),
            spirv::StorageClass::Generic,
        ),
    );
    let u32_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B32),
            spirv::StorageClass::Generic,
        ),
    );
    let shared_u8_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B8),
            spirv::StorageClass::Workgroup,
        ),
    );
    let shared_u32_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B32),
            spirv::StorageClass::Workgroup,
        ),
    );
    let src_ptr = builder.convert_u_to_ptr(u8_ptr, None, arg.src)?;
    let dst_ptr = builder.convert_u_to_ptr(shared_u8_ptr, None, arg.dst)?;
    for offset in (0..size).step_by(4) {
        let offset_const = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(offset),
        )?;
        let word = if offset == 0 {
            builder.bitcast(u32_ptr, None, src_ptr)?
        } else {
            let advanced = builder.in_bounds_ptr_access_chain(
                u8_ptr,
                None,
                src_ptr,
                offset_const,
                iter::empty(),
            )?;
            builder.bitcast(u32_ptr, None, advanced)?
        };
        let word = builder.load(
            u32_type,
            None,
            word,
            Some(spirv::MemoryAccess::ALIGNED),
            [dr::Operand::LiteralInt32(4)].iter().cloned(),
        )?;
        // A word beyond the source size is not copied but cleared.
        let word = match arg.src_size {
            Some(src_size) => {
                let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
                let zero = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(0u32),
                )?;
                let enough =
                    builder.u_greater_than(bool_type, None, src_size, offset_const)?;
                builder.select(u32_type, None, enough, word, zero)?
            }
            None => word,
        };
        let target = if offset == 0 {
            builder.bitcast(shared_u32_ptr, None, dst_ptr)?
        } else {
            let advanced = builder.in_bounds_ptr_access_chain(
                shared_u8_ptr,
                None,
                dst_ptr,
                offset_const,
                iter::empty(),
            )?;
            builder.bitcast(shared_u32_ptr, None, advanced)?
        };
        builder.store(target, word, None, iter::empty())?;
    }
    Ok(())
}

// ldmatrix.sync.aligned.m8n8.x{1,2,4}.b16 [.trans] {d...}, [p];
//
// The rows of matrix i are the addresses held by threads 8i..8i+7, and thread
// t receives the two 16-bit elements at row t/4, columns 2*(t%4) and
// 2*(t%4)+1 of every matrix, so a row address has to be fetched from its
// owner first. ".trans" distributes the transposed matrix: rows 2*(t%4) and
// 2*(t%4)+1 at column t/4.
/// `vsub4.u32.u32.u32 d, a, b, c;` - four independent byte subtractions, each
/// result byte holding the low eight bits of the difference, so the bytes do
/// not borrow into each other. The saturating variant is a different
/// instruction.
fn emit_vsub4(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let mask = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0xffu32),
    )?;
    let mut result: Option<spirv::Word> = None;
    for byte in 0u32..4 {
        let shift = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(byte * 8),
        )?;
        if byte == 0 {
            let low = builder.bitwise_and(u32_type, None, arg.src1, mask)?;
            let low_b = builder.bitwise_and(u32_type, None, arg.src2, mask)?;
            // A byte difference is negative as often as not and sets every
            // upper bit of a 32-bit register, so the byte has to be cut out
            // again.
            let low = builder.i_sub(u32_type, None, low, low_b)?;
            result = Some(builder.bitwise_and(u32_type, None, low, mask)?);
            continue;
        }
        let a = builder.shift_right_logical(u32_type, None, arg.src1, shift)?;
        let a = builder.bitwise_and(u32_type, None, a, mask)?;
        let b = builder.shift_right_logical(u32_type, None, arg.src2, shift)?;
        let b = builder.bitwise_and(u32_type, None, b, mask)?;
        let byte_diff = builder.i_sub(u32_type, None, a, b)?;
        let byte_diff = builder.bitwise_and(u32_type, None, byte_diff, mask)?;
        let byte_diff = builder.shift_left_logical(u32_type, None, byte_diff, shift)?;
        result = Some(builder.bitwise_or(u32_type, None, result.unwrap_or(0), byte_diff)?);
    }
    builder.copy_object(u32_type, Some(arg.dst), result.unwrap_or(0))?;
    Ok(())
}

/// `vset4.u32.u32.ne d, a, b, c;` - four byte comparisons, each result one or
/// zero (the `cmp` intrinsics return 0xff and are different instructions).
fn emit_vset4(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    op: &ast::CompareOp,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let bool_type = map.get_or_add_scalar(builder, ast::ScalarType::Pred);
    let mask = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0xffu32),
    )?;
    let one = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(1u32),
    )?;
    let zero = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0u32),
    )?;
    let mut result: Option<spirv::Word> = None;
    for byte in 0u32..4 {
        let shift = map.get_or_add_constant(
            builder,
            &ast::Type::Scalar(ast::ScalarType::U32),
            &vec_repr(byte * 8),
        )?;
        let a = builder.shift_right_logical(u32_type, None, arg.src1, shift)?;
        let a = builder.bitwise_and(u32_type, None, a, mask)?;
        let b = builder.shift_right_logical(u32_type, None, arg.src2, shift)?;
        let b = builder.bitwise_and(u32_type, None, b, mask)?;
        let holds = match op {
            ast::CompareOp::Eq => builder.i_equal(bool_type, None, a, b)?,
            ast::CompareOp::Ne => builder.i_not_equal(bool_type, None, a, b)?,
            ast::CompareOp::Lt => builder.u_less_than(bool_type, None, a, b)?,
            ast::CompareOp::Le => builder.u_less_than_equal(bool_type, None, a, b)?,
            ast::CompareOp::Gt => builder.u_greater_than(bool_type, None, a, b)?,
            ast::CompareOp::Ge => builder.u_greater_than_equal(bool_type, None, a, b)?,
        };
        let piece = builder.select(u32_type, None, holds, one, zero)?;
        let piece = builder.shift_left_logical(u32_type, None, piece, shift)?;
        result = Some(match result {
            None => piece,
            Some(previous) => builder.bitwise_or(u32_type, None, previous, piece)?,
        });
    }
    builder.copy_object(u32_type, Some(arg.dst), result.unwrap_or(0))?;
    Ok(())
}

/// `redux.sync.add.s32 d, a, mask;` - the value `a` reduced across the warp.
/// The member mask is a promise about which lanes take part; the operation
/// covers the whole subgroup.
fn emit_redux(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    op: &ast::ReduxOp,
    typ: &ast::ScalarType,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let result_type = map.get_or_add_scalar(builder, ast::ScalarType::from(*typ));
    let subgroup = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(spirv::Scope::Subgroup as u32),
    )?;
    let value = arg.src1;
    let reduce = spirv::GroupOperation::Reduce;
    let result = match (op, typ) {
        (ast::ReduxOp::Add, ast::ScalarType::F32) => {
            builder.group_non_uniform_f_add(result_type, None, subgroup, reduce, value, None)?
        }

        (ast::ReduxOp::Min, ast::ScalarType::F32) => {
            builder.group_non_uniform_f_min(result_type, None, subgroup, reduce, value, None)?
        }
        (ast::ReduxOp::Max, ast::ScalarType::F32) => {
            builder.group_non_uniform_f_max(result_type, None, subgroup, reduce, value, None)?
        }
        (ast::ReduxOp::Add, _) => builder.group_non_uniform_i_add(
            result_type,
            None,
            subgroup,
            reduce,
            value,
            None,
        )?,
        // The bitwise reductions and the 64-bit float minimum and maximum have
        // group operations of their own in SPIR-V 1.3; only the 16-bit and
        // packed float forms are left unimplemented.
        (ast::ReduxOp::And, _) => builder.group_non_uniform_bitwise_and(
            result_type,
            None,
            subgroup,
            reduce,
            value,
            None,
        )?,
        (ast::ReduxOp::Or, _) => builder.group_non_uniform_bitwise_or(
            result_type,
            None,
            subgroup,
            reduce,
            value,
            None,
        )?,
        (ast::ReduxOp::Xor, _) => builder.group_non_uniform_bitwise_xor(
            result_type,
            None,
            subgroup,
            reduce,
            value,
            None,
        )?,
        (ast::ReduxOp::Min, ast::ScalarType::F64) => {
            builder.group_non_uniform_f_min(result_type, None, subgroup, reduce, value, None)?
        }
        (ast::ReduxOp::Max, ast::ScalarType::F64) => {
            builder.group_non_uniform_f_max(result_type, None, subgroup, reduce, value, None)?
        }
        (ast::ReduxOp::Add, ast::ScalarType::F64) => {
            builder.group_non_uniform_f_add(result_type, None, subgroup, reduce, value, None)?
        }
        _ => return Err(error_todo()),
    };
    builder.copy_object(result_type, Some(arg.dst), result)?;
    Ok(())
}

/// `cvt.rn.f16x2.f32 d, a, b;` and its bfloat counterpart: two floats are
/// converted and packed into one register, the first one in the low half.
fn emit_cvt_packed(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    typ: ast::ScalarType,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let b32_type = map.get_or_add_scalar(builder, ast::ScalarType::B32);
    let u16_type = map.get_or_add_scalar(builder, ast::ScalarType::U16);
    let f32_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
    let sixteen = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(16u32),
    )?;
    let convert = |builder: &mut dr::Builder,
                   map: &mut TypeWordMap,
                   value: spirv::Word|
     -> Result<spirv::Word, TranslateError> {
        if typ == ast::ScalarType::B16 {
            // bfloat16 keeps the upper half of the float, rounded to nearest
            // even.
            let bits = builder.bitcast(u32_type, None, value)?;
            let rounding = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(0x7fffu32),
            )?;
            // Round to nearest even: half of what is dropped, plus the low bit
            // of the part that is kept so that a tie goes to the even value.
            let one = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(1u32),
            )?;
            let kept = builder.shift_right_logical(u32_type, None, bits, sixteen)?;
            let lsb = builder.bitwise_and(u32_type, None, kept, one)?;
            let biased = builder.i_add(u32_type, None, bits, rounding)?;
            let biased = builder.i_add(u32_type, None, biased, lsb)?;
            return Ok(builder.shift_right_logical(u32_type, None, biased, sixteen)?);
        }
        let f16_type = map.get_or_add_scalar(builder, ast::ScalarType::F16);
        let half = builder.f_convert(f16_type, None, value)?;
        let half = builder.bitcast(u16_type, None, half)?;
        Ok(builder.u_convert(u32_type, None, half)?)
    };
    // "cvt.rn.f16x2.f32 d, a, b" puts a in the HIGH half and b in the LOW
    // half. Getting it backwards silently exchanges every adjacent pair of
    // elements in whatever the caller stages.
    let high = convert(builder, map, arg.src1)?;
    let low = convert(builder, map, arg.src2)?;
    let high = builder.shift_left_logical(u32_type, None, high, sixteen)?;
    let joined = builder.bitwise_or(u32_type, None, low, high)?;
    let joined = builder.bitcast(b32_type, None, joined)?;
    builder.copy_object(b32_type, Some(arg.dst), joined)?;
    Ok(())
}

// movmatrix.sync.aligned.m8n8.trans.b16 d, a;
//
// The warp holds an 8x8 matrix of 16-bit values, one register per thread:
// thread t holds row t/4, columns 2*(t%4) and 2*(t%4)+1. The transposed matrix
// has the same distribution, so t's result is row 2*(t%4) column t/4, followed
// by row 2*(t%4)+1 at the same column.
fn emit_movmatrix(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    laneid: Option<spirv::Word>,
    arg: &ast::Arg2<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let laneid = laneid.ok_or(error_todo())?;
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let b32_type = map.get_or_add_scalar(builder, ast::ScalarType::B32);
    let two = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(2u32),
    )?;
    let three = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(3u32),
    )?;
    let four = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(4u32),
    )?;
    let one = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(1u32),
    )?;
    let sixteen = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(16u32),
    )?;
    let mask = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(0xffffu32),
    )?;
    let lane = builder.load(u32_type, None, laneid, None, [].iter().cloned())?;
    // column = lane / 4, row pair = 2 * (lane % 4)
    let column = builder.shift_right_logical(u32_type, None, lane, two)?;
    let in_group = builder.bitwise_and(u32_type, None, lane, three)?;
    let low_row = builder.i_add(u32_type, None, in_group, in_group)?;
    let one_row = builder.i_add(u32_type, None, low_row, one.clone())?;
    // The source thread of a value at (row, column) is 4 * row + column / 2,
    // and the half it holds is column % 2.
    let column_pair = builder.shift_right_logical(u32_type, None, column, one.clone())?;
    let low_lane_base = builder.shift_left_logical(u32_type, None, low_row, two)?;
    let low_source = builder.i_add(u32_type, None, low_lane_base, column_pair)?;
    let high_lane_base = builder.shift_left_logical(u32_type, None, one_row, two)?;
    let high_source = builder.i_add(u32_type, None, high_lane_base, column_pair)?;
    let half = builder.bitwise_and(u32_type, None, column, one)?;
    let shift = builder.shift_left_logical(u32_type, None, half, four)?;
    let low_packed = builder.subgroup_shuffle_intel(b32_type, None, arg.src, low_source)?;
    let high_packed = builder.subgroup_shuffle_intel(b32_type, None, arg.src, high_source)?;
    let low_packed = builder.bitcast(u32_type, None, low_packed)?;
    let high_packed = builder.bitcast(u32_type, None, high_packed)?;
    let low = builder.shift_right_logical(u32_type, None, low_packed, shift)?;
    let high = builder.shift_right_logical(u32_type, None, high_packed, shift)?;
    let low = builder.bitwise_and(u32_type, None, low, mask)?;
    let high = builder.bitwise_and(u32_type, None, high, mask)?;
    let high = builder.shift_left_logical(u32_type, None, high, sixteen)?;
    let joined = builder.bitwise_or(u32_type, None, low, high)?;
    let result = builder.bitcast(b32_type, None, joined)?;
    builder.copy_object(b32_type, Some(arg.dst), result)?;
    Ok(())
}

fn emit_ldmatrix(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    laneid: Option<spirv::Word>,
    details: &ast::LdMatrixDetails,
    arg: &ast::Arg2LdMatrix<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let laneid = laneid.ok_or(error_todo())?;
    let num = details.num as usize;
    if num == 0 || num > 4 {
        return Err(error_todo());
    }
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let u64_type = map.get_or_add_scalar(builder, ast::ScalarType::U64);
    let u16_type = map.get_or_add_scalar(builder, ast::ScalarType::U16);
    let u8_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B8),
            spirv::StorageClass::Generic,
        ),
    );
    let u32_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B32),
            spirv::StorageClass::Generic,
        ),
    );
    let u16_ptr = map.get_or_add(
        builder,
        SpirvType::pointer_to(
            ast::Type::Scalar(ast::ScalarType::B16),
            spirv::StorageClass::Generic,
        ),
    );
    let c1 = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(1u32),
    )?;
    let c2 = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(2u32),
    )?;
    let c3 = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(3u32),
    )?;
    let lane = builder.load(u32_type, None, laneid, None, [].iter().cloned())?;
    // %lane / 4 and %lane % 4 select the row fragment and the pair of
    // elements inside the row.
    let lane_div4 = builder.shift_right_logical(u32_type, None, lane, c2)?;
    let lane_mod4 = builder.bitwise_and(u32_type, None, lane, c3)?;
    let mut fragments = Vec::with_capacity(num);
    for i in 0..num {
        let base = (8 * i) as u32;
        let fragment = if details.trans {
            // Row 2*(t%4) and 2*(t%4)+1, column t/4.
            let mut doubles = builder.shift_left_logical(u32_type, None, lane_mod4, c1)?;
            if base != 0 {
                let offset = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(base),
                )?;
                doubles = builder.i_add(u32_type, None, doubles, offset)?;
            }
            let next = builder.i_add(u32_type, None, doubles, c1)?;
            let byte_offset = builder.shift_left_logical(u32_type, None, lane_div4, c1)?;
            let byte_offset = builder.u_convert(u64_type, None, byte_offset)?;
            let low = ldmatrix_load(
                builder, map, arg.src, doubles, byte_offset, u64_type, u8_ptr, u16_ptr, u16_type,
                2,
            )?;
            let low = builder.u_convert(u32_type, None, low)?;
            let high = ldmatrix_load(
                builder, map, arg.src, next, byte_offset, u64_type, u8_ptr, u16_ptr, u16_type, 2,
            )?;
            let high = builder.u_convert(u32_type, None, high)?;
            let shift = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(16u32),
            )?;
            let high = builder.shift_left_logical(u32_type, None, high, shift)?;
            builder.bitwise_or(u32_type, None, low, high)?
        } else {
            // Row t/4, columns 2*(t%4) and 2*(t%4)+1.
            let mut src_lane = lane_div4;
            if base != 0 {
                let offset = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(base),
                )?;
                src_lane = builder.i_add(u32_type, None, src_lane, offset)?;
            }
            let byte_offset = builder.shift_left_logical(u32_type, None, lane_mod4, c2)?;
            let byte_offset = builder.u_convert(u64_type, None, byte_offset)?;
            ldmatrix_load(
                builder, map, arg.src, src_lane, byte_offset, u64_type, u8_ptr, u32_ptr, u32_type,
                4,
            )?
        };
        fragments.push(fragment);
    }
    if num == 1 {
        // A single fragment register holds one 8x8 tile; there is no SPIR-V
        // vector of one element.
        let b32_type = map.get_or_add_scalar(builder, ast::ScalarType::B32);
        builder.copy_object(b32_type, Some(arg.dst), fragments[0])?;
    } else {
        let vector_type =
            map.get_or_add(builder, SpirvType::Vector(SpirvScalarKey::B32, num as u8));
        builder.composite_construct(vector_type, Some(arg.dst), fragments.into_iter())?;
    }
    Ok(())
}

/// Read the elements of one matrix row from shared memory: the row address
/// is held by `src_lane`, so it is fetched from there first.
#[allow(clippy::too_many_arguments)]
fn ldmatrix_load(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    address: spirv::Word,
    src_lane: spirv::Word,
    byte_offset: spirv::Word,
    u64_type: spirv::Word,
    u8_ptr: spirv::Word,
    element_ptr: spirv::Word,
    result_type: spirv::Word,
    element_size: u32,
) -> Result<spirv::Word, TranslateError> {
    // The shuffle is a 32-bit operation, so the 64-bit address travels as
    // two halves.
    let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
    let b32_type = map.get_or_add_scalar(builder, ast::ScalarType::B32);
    // The linker of the OpenCL runtime requires the shift amount to have the
    // same type as the shifted value.
    let thirty_two = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(32u32),
    )?;
    let thirty_two_wide = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U64),
        &vec_repr(32u64),
    )?;
    let low = builder.u_convert(u32_type, None, address)?;
    let high = builder.shift_right_logical(u64_type, None, address, thirty_two_wide)?;
    let high = builder.u_convert(u32_type, None, high)?;
    let low = builder.subgroup_shuffle_intel(b32_type, None, low, src_lane)?;
    let high = builder.subgroup_shuffle_intel(b32_type, None, high, src_lane)?;
    let row = builder.u_convert(u64_type, None, high)?;
    let row = builder.shift_left_logical(u64_type, None, row, thirty_two_wide)?;
    let low = builder.u_convert(u64_type, None, low)?;
    let row = builder.bitwise_or(u64_type, None, row, low)?;
    let address = builder.i_add(u64_type, None, row, byte_offset)?;
    let pointer = builder.convert_u_to_ptr(u8_ptr, None, address)?;
    let pointer = builder.bitcast(element_ptr, None, pointer)?;
    Ok(builder.load(
        result_type,
        None,
        pointer,
        Some(spirv::MemoryAccess::ALIGNED),
        [dr::Operand::LiteralInt32(element_size)].iter().cloned(),
    )?)
}

fn emit_sqrt(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    details: &ast::SqrtDetails,
    a: &ast::Arg2<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let result_type = map.get_or_add_scalar(builder, details.typ.into());
    let (ocl_op, rounding) = match details.kind {
        ast::SqrtKind::Approx => (spirv::CLOp::native_sqrt, None),
        ast::SqrtKind::Rounding(rnd) => (spirv::CLOp::sqrt, Some(rnd)),
    };
    builder.ext_inst(
        result_type,
        Some(a.dst),
        opencl,
        ocl_op as spirv::Word,
        [dr::Operand::IdRef(a.src)].iter().cloned(),
    )?;
    let _ = rounding;
    Ok(())
}

fn emit_float_div_decoration(builder: &mut dr::Builder, dst: spirv::Word, kind: ast::DivFloatKind) {
    match kind {
        ast::DivFloatKind::Approx => {
            builder.decorate(
                dst,
                spirv::Decoration::FPFastMathMode,
                [dr::Operand::FPFastMathMode(
                    spirv::FPFastMathMode::ALLOW_RECIP,
                )]
                .iter()
                .cloned(),
            );
        }
        ast::DivFloatKind::Rounding(_rnd) => {}
        ast::DivFloatKind::Full => {}
    }
}

fn emit_atom(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    details: &ast::AtomDetails,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let (spirv_op, typ) = match details.inner {
        ast::AtomInnerDetails::Bit { op, typ } => {
            let spirv_op = match op {
                ast::AtomBitOp::And => dr::Builder::atomic_and,
                ast::AtomBitOp::Or => dr::Builder::atomic_or,
                ast::AtomBitOp::Xor => dr::Builder::atomic_xor,
                ast::AtomBitOp::Exchange => dr::Builder::atomic_exchange,
            };
            (spirv_op, ast::ScalarType::from(typ))
        }
        ast::AtomInnerDetails::Unsigned { op, typ } => {
            let spirv_op = match op {
                ast::AtomUIntOp::Add => dr::Builder::atomic_i_add,
                ast::AtomUIntOp::Inc | ast::AtomUIntOp::Dec => {
                    return Err(error_unreachable());
                }
                ast::AtomUIntOp::Min => dr::Builder::atomic_u_min,
                ast::AtomUIntOp::Max => dr::Builder::atomic_u_max,
            };
            (spirv_op, typ.into())
        }
        ast::AtomInnerDetails::Signed { op, typ } => {
            let spirv_op = match op {
                ast::AtomSIntOp::Add => dr::Builder::atomic_i_add,
                ast::AtomSIntOp::Min => dr::Builder::atomic_s_min,
                ast::AtomSIntOp::Max => dr::Builder::atomic_s_max,
            };
            (spirv_op, typ.into())
        }
        ast::AtomInnerDetails::Float { op, typ } => {
            let spirv_op: fn(&mut dr::Builder, _, _, _, _, _, _) -> _ = match op {
                ast::AtomFloatOp::Add => dr::Builder::atomic_f_add_ext,
            };
            (spirv_op, typ.into())
        }
    };
    let result_type = map.get_or_add_scalar(builder, typ);
    let src_ptr = arg.src1;
    let memory_const = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(details.scope.to_spirv() as u32),
    )?;
    let semantics_const = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(details.semantics.to_spirv().bits()),
    )?;
    spirv_op(
        builder,
        result_type,
        Some(arg.dst),
        src_ptr,
        memory_const,
        semantics_const,
        arg.src2,
    )?;
    Ok(())
}

#[derive(Clone)]
struct PtxImplImport {
    out_arg: ast::Type,
    fn_id: u32,
    in_args: Vec<ast::Type>,
}

/// PTX ".sat" on a float arithmetic result clamps it to [0, 1].
fn saturate_float_result(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    value: spirv::Word,
    typ: ast::ScalarType,
) -> Result<spirv::Word, dr::Error> {
    let clamp = |builder: &mut dr::Builder,
                 result_type: spirv::Word|
     -> Result<spirv::Word, dr::Error> {
        let zero = builder.constant_f32(result_type, None, 0.0);
        let one = builder.constant_f32(result_type, None, 1.0);
        let clamped = builder.id();
        builder.ext_inst(
            result_type,
            Some(clamped),
            opencl,
            spirv::CLOp::fclamp as u32,
            [
                dr::Operand::IdRef(value),
                dr::Operand::IdRef(zero),
                dr::Operand::IdRef(one),
            ]
            .iter()
            .cloned(),
        )?;
        Ok(clamped)
    };
    match typ {
        ast::ScalarType::F32 => {
            let result_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
            clamp(builder, result_type)
        }
        ast::ScalarType::F16 => {
            // Clamp in single precision, which needs no half constant.
            let f32_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
            let wide = builder.f_convert(f32_type, None, value)?;
            let clamped = clamp(builder, f32_type)?;
            let half_type = map.get_or_add_scalar(builder, ast::ScalarType::F16);
            Ok(builder.f_convert(half_type, None, clamped)?)
        }
        _ => Ok(value),
    }
}

fn emit_mul_float(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    ctr: &ast::ArithFloat,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    // A packed pair of halves is a vector of two halves, not a scalar.
    let result_type = map.get_or_add(builder, SpirvType::from(ctr.typ));
    if needs_wide_rounding(ast::Type::Scalar(ctr.typ), ctr.rounding) {
        return emit_rounded_float(
            builder,
            map,
            opencl,
            RoundedFloatOp::Mul,
            ctr.rounding,
            arg.dst,
            arg.src1,
            arg.src2,
        );
    }
    if ctr.saturate {
        let product = builder.f_mul(result_type, None, arg.src1, arg.src2)?;
        let saturated = saturate_float_result(builder, map, opencl, product, ctr.typ)?;
        builder.copy_object(result_type, Some(arg.dst), saturated)?;
        return Ok(());
    }
    builder.f_mul(result_type, Some(arg.dst), arg.src1, arg.src2)?;
    Ok(())
}

fn emit_rcp(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    desc: &ast::RcpDetails,
    a: &ast::Arg2<ExpandedArgParams>,
) -> Result<(), TranslateError> {
    let (instr_type, constant) = if desc.is_f64 {
        (ast::ScalarType::F64, vec_repr(1.0f64))
    } else {
        (ast::ScalarType::F32, vec_repr(1.0f32))
    };
    let one = map.get_or_add_constant(builder, &ast::Type::Scalar(instr_type), &constant)?;
    let result_type = map.get_or_add_scalar(builder, instr_type);
    builder.f_div(result_type, Some(a.dst), one, a.src)?;
    if desc.rounding.is_none() {
        // Only the approximate form may be compiled as a reciprocal estimate;
        // rcp.rn promises the correctly rounded quotient.
        builder.decorate(
            a.dst,
            spirv::Decoration::FPFastMathMode,
            [dr::Operand::FPFastMathMode(
                spirv::FPFastMathMode::ALLOW_RECIP,
            )]
            .iter()
            .cloned(),
        );
    }
    Ok(())
}

fn vec_repr<T: Copy>(t: T) -> Vec<u8> {
    let mut result = vec![0; mem::size_of::<T>()];
    unsafe { std::ptr::copy_nonoverlapping(&t, result.as_mut_ptr() as *mut _, 1) };
    result
}

fn emit_variable(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    var: &ast::Variable<spirv::Word>,
) -> Result<(), TranslateError> {
    let (must_init, st_class) = match var.state_space {
        ast::StateSpace::Reg | ast::StateSpace::Param | ast::StateSpace::Local => {
            (false, spirv::StorageClass::Function)
        }
        ast::StateSpace::Global => (true, spirv::StorageClass::CrossWorkgroup),
        ast::StateSpace::Shared => (false, spirv::StorageClass::Workgroup),
        ast::StateSpace::Const => todo!(),
        ast::StateSpace::Generic => todo!(),
        ast::StateSpace::Sreg => todo!(),
    };
    let initalizer = if var.array_init.len() > 0 {
        Some(map.get_or_add_constant(
            builder,
            &ast::Type::from(var.v_type.clone()),
            &*var.array_init,
        )?)
    } else if must_init {
        let type_id = map.get_or_add(builder, SpirvType::new(var.v_type.clone()));
        Some(builder.constant_null(type_id, None))
    } else {
        None
    };
    let ptr_type_id = map.get_or_add(builder, SpirvType::pointer_to(var.v_type.clone(), st_class));
    builder.variable(ptr_type_id, Some(var.name), st_class, initalizer);
    if let Some(align) = var.align {
        builder.decorate(
            var.name,
            spirv::Decoration::Alignment,
            [dr::Operand::LiteralInt32(align)].iter().cloned(),
        );
    }
    Ok(())
}

fn emit_mad_uint(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::MulUInt,
    arg: &ast::Arg4<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    match desc.control {
        ast::MulIntControl::Low => {
            let mul_result = builder.i_mul(inst_type, None, arg.src1, arg.src2)?;
            builder.i_add(inst_type, Some(arg.dst), arg.src3, mul_result)?;
        }
        ast::MulIntControl::High => {
            builder.ext_inst(
                inst_type,
                Some(arg.dst),
                opencl,
                spirv::CLOp::u_mad_hi as spirv::Word,
                [
                    dr::Operand::IdRef(arg.src1),
                    dr::Operand::IdRef(arg.src2),
                    dr::Operand::IdRef(arg.src3),
                ]
                .iter()
                .cloned(),
            )?;
        }
        ast::MulIntControl::Wide => {
            // mad.wide.u32 d, a, b, c: the product is 64 bits wide and c is
            // added to it without truncation.
            let wide_type = map.get_or_add_scalar(builder, ast::ScalarType::U64);
            let a = builder.u_convert(wide_type, None, arg.src1)?;
            let b = builder.u_convert(wide_type, None, arg.src2)?;
            let product = builder.i_mul(wide_type, None, a, b)?;
            let c = builder.u_convert(wide_type, None, arg.src3)?;
            builder.i_add(wide_type, Some(arg.dst), product, c)?;
        }
    };
    Ok(())
}

fn emit_mad_sint(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::MulSInt,
    arg: &ast::Arg4<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    match desc.control {
        ast::MulIntControl::Low => {
            let mul_result = builder.i_mul(inst_type, None, arg.src1, arg.src2)?;
            builder.i_add(inst_type, Some(arg.dst), arg.src3, mul_result)?;
        }
        ast::MulIntControl::High => {
            builder.ext_inst(
                inst_type,
                Some(arg.dst),
                opencl,
                spirv::CLOp::s_mad_hi as spirv::Word,
                [
                    dr::Operand::IdRef(arg.src1),
                    dr::Operand::IdRef(arg.src2),
                    dr::Operand::IdRef(arg.src3),
                ]
                .iter()
                .cloned(),
            )?;
        }
        ast::MulIntControl::Wide => {
            let wide_type = map.get_or_add_scalar(builder, ast::ScalarType::S64);
            let a = builder.s_convert(wide_type, None, arg.src1)?;
            let b = builder.s_convert(wide_type, None, arg.src2)?;
            let product = builder.i_mul(wide_type, None, a, b)?;
            let c = builder.s_convert(wide_type, None, arg.src3)?;
            builder.i_add(wide_type, Some(arg.dst), product, c)?;
        }
    };
    Ok(())
}

fn emit_mad_float(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::ArithFloat,
    arg: &ast::Arg4<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    if needs_wide_rounding(ast::Type::Scalar(desc.typ), desc.rounding) {
        return emit_rounded_float(
            builder,
            map,
            opencl,
            RoundedFloatOp::Mad(arg.src3),
            desc.rounding,
            arg.dst,
            arg.src1,
            arg.src2,
        );
    }
    if desc.saturate {
        let product = builder.ext_inst(
            inst_type,
            None,
            opencl,
            spirv::CLOp::fma as spirv::Word,
            [
                dr::Operand::IdRef(arg.src1),
                dr::Operand::IdRef(arg.src2),
                dr::Operand::IdRef(arg.src3),
            ]
            .iter()
            .cloned(),
        )?;
        let saturated = saturate_float_result(builder, map, opencl, product, desc.typ)?;
        builder.copy_object(inst_type, Some(arg.dst), saturated)?;
        return Ok(());
    }
    builder.ext_inst(
        inst_type,
        Some(arg.dst),
        opencl,
        spirv::CLOp::mad as spirv::Word,
        [
            dr::Operand::IdRef(arg.src1),
            dr::Operand::IdRef(arg.src2),
            dr::Operand::IdRef(arg.src3),
        ]
        .iter()
        .cloned(),
    )?;
    Ok(())
}

fn emit_add_float(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::ArithFloat,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    if needs_wide_rounding(ast::Type::Scalar(desc.typ), desc.rounding) {
        return emit_rounded_float(
            builder,
            map,
            opencl,
            RoundedFloatOp::Add,
            desc.rounding,
            arg.dst,
            arg.src1,
            arg.src2,
        );
    }
    if desc.saturate {
        let add = builder.f_add(inst_type, None, arg.src1, arg.src2)?;
        let saturated = saturate_float_result(builder, map, opencl, add, desc.typ)?;
        builder.copy_object(inst_type, Some(arg.dst), saturated)?;
        return Ok(());
    }
    builder.f_add(inst_type, Some(arg.dst), arg.src1, arg.src2)?;
    Ok(())
}

fn emit_sub_float(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::ArithFloat,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    if needs_wide_rounding(ast::Type::Scalar(desc.typ), desc.rounding) {
        return emit_rounded_float(
            builder,
            map,
            opencl,
            RoundedFloatOp::Sub,
            desc.rounding,
            arg.dst,
            arg.src1,
            arg.src2,
        );
    }
    if desc.saturate {
        let sub = builder.f_sub(inst_type, None, arg.src1, arg.src2)?;
        let saturated = saturate_float_result(builder, map, opencl, sub, desc.typ)?;
        builder.copy_object(inst_type, Some(arg.dst), saturated)?;
        return Ok(());
    }
    builder.f_sub(inst_type, Some(arg.dst), arg.src1, arg.src2)?;
    Ok(())
}

fn emit_min(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::MinMaxDetails,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let cl_op = match desc {
        ast::MinMaxDetails::Signed(_) => spirv::CLOp::s_min,
        ast::MinMaxDetails::Unsigned(_) => spirv::CLOp::u_min,
        ast::MinMaxDetails::Float(_) => spirv::CLOp::fmin,
    };
    let inst_type = map.get_or_add(builder, SpirvType::new(desc.get_type()));
    builder.ext_inst(
        inst_type,
        Some(arg.dst),
        opencl,
        cl_op as spirv::Word,
        [dr::Operand::IdRef(arg.src1), dr::Operand::IdRef(arg.src2)]
            .iter()
            .cloned(),
    )?;
    Ok(())
}

fn emit_max(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::MinMaxDetails,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let cl_op = match desc {
        ast::MinMaxDetails::Signed(_) => spirv::CLOp::s_max,
        ast::MinMaxDetails::Unsigned(_) => spirv::CLOp::u_max,
        ast::MinMaxDetails::Float(_) => spirv::CLOp::fmax,
    };
    let inst_type = map.get_or_add(builder, SpirvType::new(desc.get_type()));
    builder.ext_inst(
        inst_type,
        Some(arg.dst),
        opencl,
        cl_op as spirv::Word,
        [dr::Operand::IdRef(arg.src1), dr::Operand::IdRef(arg.src2)]
            .iter()
            .cloned(),
    )?;
    Ok(())
}

fn emit_cvt(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    dets: &ast::CvtDetails,
    arg: &ast::Arg2<ExpandedArgParams>,
    ) -> Result<(), TranslateError> {
    match dets {
        ast::CvtDetails::FloatFromFloat(desc) => {
            // bfloat16 has no SPIR-V type: the conversion is done on the bit
            // pattern. A bf16 value is the upper 16 bits of an f32.
            if desc.dst == ast::ScalarType::B16 && desc.src == ast::ScalarType::F32 {
                // f32 -> bf16: round to nearest even, then keep the upper half.
                let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                let bits = builder.bitcast(u32_type, None, arg.src)?;
                let rounding = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(0x7fffu32),
                )?;
                // Round to nearest even: the bias is half of the part that is
                // dropped plus the low bit of the kept part, so a tie goes to
                // the even value.
                let shift = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(16u32),
                )?;
                let one = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(1u32),
                )?;
                let kept = builder.shift_right_logical(u32_type, None, bits, shift)?;
                let lsb = builder.bitwise_and(u32_type, None, kept, one)?;
                let biased = builder.i_add(u32_type, None, bits, rounding)?;
                let biased = builder.i_add(u32_type, None, biased, lsb)?;
                let high = builder.shift_right_logical(u32_type, None, biased, shift)?;
                let b16_type = map.get_or_add_scalar(builder, ast::ScalarType::B16);
                let high = builder.u_convert(b16_type, None, high)?;
                builder.copy_object(b16_type, Some(arg.dst), high)?;
                return Ok(());
            }
            if desc.dst == ast::ScalarType::F32 && desc.src == ast::ScalarType::B16 {
                // bf16 -> f32: the value is the upper half of an f32.
                let u32_type = map.get_or_add_scalar(builder, ast::ScalarType::U32);
                let b16_type = map.get_or_add_scalar(builder, ast::ScalarType::B16);
                let bits = builder.u_convert(u32_type, None, arg.src)?;
                let shift = map.get_or_add_constant(
                    builder,
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    &vec_repr(16u32),
                )?;
                let bits = builder.shift_left_logical(u32_type, None, bits, shift)?;
                let f32_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
                let value = builder.bitcast(f32_type, None, bits)?;
                builder.copy_object(f32_type, Some(arg.dst), value)?;
                return Ok(());
            }
            if desc.saturate {
                if desc.dst == desc.src {
                    let ty_id = map.get_or_add(builder, SpirvType::from(desc.dst));
                    let zero = builder.constant_f32(ty_id, None, 0.0);
                    let one = builder.constant_f32(ty_id, None, 1.0);
                    let dest_t: ast::ScalarType = desc.dst.into();
                    let result_type = map.get_or_add(builder, SpirvType::from(dest_t));
                    let clamped = builder.id();
                    builder.ext_inst(
                        result_type,
                        Some(clamped),
                        opencl,
                        spirv::CLOp::fclamp as u32,
                        [
                            dr::Operand::IdRef(arg.src),
                            dr::Operand::IdRef(zero),
                            dr::Operand::IdRef(one),
                        ]
                        .iter()
                        .cloned(),
                    )?;
                    builder.copy_object(result_type, Some(arg.dst), clamped)?;
                    if std::env::var_os("ZLUDA_PTX_DISABLE_SAT").is_some() {
                        return Ok(());
                    }
                } else {
                    // f64->f32 or f32->f64 with saturate: clamp [0,1] then convert
                    let dest_t: ast::ScalarType = desc.dst.into();
                    let src_t: ast::ScalarType = desc.src.into();
                    let src_type = map.get_or_add(builder, SpirvType::from(src_t));
                    let zero = builder.constant_f64(src_type, None, 0.0);
                    let one = builder.constant_f64(src_type, None, 1.0);
                    let clamped = builder.id();
                    builder.ext_inst(
                        src_type,
                        Some(clamped),
                        opencl,
                        spirv::CLOp::fclamp as u32,
                        [
                            dr::Operand::IdRef(arg.src),
                            dr::Operand::IdRef(zero),
                            dr::Operand::IdRef(one),
                        ]
                        .iter()
                        .cloned(),
                    )?;
                    let result_type = map.get_or_add(builder, SpirvType::from(dest_t));
                    builder.f_convert(result_type, Some(arg.dst), clamped)?;
                    emit_rounding_decoration(builder, arg.dst, desc.rounding);
                }
            } else {
                let dest_t: ast::ScalarType = desc.dst.into();
                let result_type = map.get_or_add(builder, SpirvType::from(dest_t));
                if desc.dst == desc.src {
                    match desc.rounding {
                        Some(ast::RoundingMode::NearestEven) => {
                            builder.ext_inst(
                                result_type,
                                Some(arg.dst),
                                opencl,
                                spirv::CLOp::rint as u32,
                                [dr::Operand::IdRef(arg.src)].iter().cloned(),
                            )?;
                        }
                        Some(ast::RoundingMode::Zero) => {
                            builder.ext_inst(
                                result_type,
                                Some(arg.dst),
                                opencl,
                                spirv::CLOp::trunc as u32,
                                [dr::Operand::IdRef(arg.src)].iter().cloned(),
                            )?;
                        }
                        Some(ast::RoundingMode::NegativeInf) => {
                            builder.ext_inst(
                                result_type,
                                Some(arg.dst),
                                opencl,
                                spirv::CLOp::floor as u32,
                                [dr::Operand::IdRef(arg.src)].iter().cloned(),
                            )?;
                        }
                        Some(ast::RoundingMode::PositiveInf) => {
                            builder.ext_inst(
                                result_type,
                                Some(arg.dst),
                                opencl,
                                spirv::CLOp::ceil as u32,
                                [dr::Operand::IdRef(arg.src)].iter().cloned(),
                            )?;
                        }
                        None => {
                            builder.copy_object(result_type, Some(arg.dst), arg.src)?;
                        }
                    }
                } else {
                    let converted = builder.f_convert(result_type, None, arg.src)?;
                    emit_rounding_decoration(builder, converted, desc.rounding);
                    builder.copy_object(result_type, Some(arg.dst), converted)?;
                }
            }
        }
        ast::CvtDetails::FloatFromInt(desc) => {
            let dest_t: ast::ScalarType = desc.dst.into();
            let result_type = map.get_or_add(builder, SpirvType::from(dest_t));
            let converted = if desc.src.kind() == ast::ScalarKind::Signed {
                builder.convert_s_to_f(result_type, None, arg.src)?
            } else {
                builder.convert_u_to_f(result_type, None, arg.src)?
            };
            emit_rounding_decoration(builder, converted, desc.rounding);
            if desc.saturate && dest_t == ast::ScalarType::F32 {
                // "cvt.sat.f32.s32" clamps the result to [0, 1].
                let zero = builder.constant_f32(result_type, None, 0.0);
                let one = builder.constant_f32(result_type, None, 1.0);
                builder.ext_inst(
                    result_type,
                    Some(arg.dst),
                    opencl,
                    spirv::CLOp::fclamp as u32,
                    [
                        dr::Operand::IdRef(converted),
                        dr::Operand::IdRef(zero),
                        dr::Operand::IdRef(one),
                    ]
                    .iter()
                    .cloned(),
                )?;
            } else {
                builder.copy_object(result_type, Some(arg.dst), converted)?;
            }
        }
        ast::CvtDetails::IntFromFloat(desc) => {
            let dest_t: ast::ScalarType = desc.dst.into();
            let result_type = map.get_or_add(builder, SpirvType::from(dest_t));
            let converted = if desc.dst.kind() == ast::ScalarKind::Signed {
                builder.convert_f_to_s(result_type, None, arg.src)?
            } else {
                builder.convert_f_to_u(result_type, None, arg.src)?
            };
            emit_rounding_decoration(builder, converted, desc.rounding);
            emit_saturating_decoration(builder, converted, desc.saturate);
            builder.copy_object(result_type, Some(arg.dst), converted)?;
        }
        ast::CvtDetails::IntFromInt(desc) => {
            let dest_t: ast::ScalarType = desc.dst.into();
            let src_t: ast::ScalarType = desc.src.into();
            let src = if desc.dst.size_of() != desc.src.size_of() {
                let new_dst = if dest_t.kind() == src_t.kind() {
                    arg.dst
                } else {
                    builder.id()
                };
                let cv = ImplicitConversion {
                    src: arg.src,
                    dst: new_dst,
                    from_type: ast::Type::Scalar(src_t),
                    from_space: ast::StateSpace::Reg,
                    to_type: ast::Type::Scalar(ast::ScalarType::from_parts(
                        dest_t.size_of(),
                        src_t.kind(),
                    )),
                    to_space: ast::StateSpace::Reg,
                    kind: ConversionKind::Default,
                };
                emit_implicit_conversion(builder, map, &cv)?;
                new_dst
            } else {
                arg.src
            };
            if dest_t.kind() == src_t.kind() {
                return Ok(());
            }
            let result_type = map.get_or_add(builder, SpirvType::from(dest_t));
            if desc.saturate {
                if desc.dst.kind() == ast::ScalarKind::Signed {
                    builder.sat_convert_u_to_s(result_type, Some(arg.dst), src)?;
                } else {
                    builder.sat_convert_s_to_u(result_type, Some(arg.dst), src)?;
                }
            } else {
                builder.bitcast(result_type, Some(arg.dst), src)?;
            }
        }
    }
    Ok(())
}

/// The arithmetic operations that PTX allows a rounding mode on.
enum RoundedFloatOp {
    Add,
    Sub,
    Mul,
    /// A fused multiply-add, with the addend.
    Mad(spirv::Word),
}

/// Whether a floating-point operation has to be evaluated in double precision
/// to honour its rounding mode.
///
/// SPIR-V has no rounding-mode operand on the arithmetic instructions, only the
/// FPRoundingMode decoration on a result, and the backend honours that on
/// conversions but ignores it elsewhere: an `add.rz` would come out rounded to
/// nearest. Double precision is exact for the product of two floats and for a
/// product plus addend whose exponents are within twenty-nine, so only the
/// conversion back needs the mode. A double operation with a mode of its own
/// has no wider type to go to and keeps nearest-even; nothing emits one.
fn needs_wide_rounding(typ: ast::Type, rounding: Option<ast::RoundingMode>) -> bool {
    match rounding {
        None | Some(ast::RoundingMode::NearestEven) => false,
        Some(_) => matches!(typ, ast::Type::Scalar(ast::ScalarType::F32)),
    }
}

fn emit_rounded_float(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    op: RoundedFloatOp,
    rounding: Option<ast::RoundingMode>,
    dst: spirv::Word,
    src1: spirv::Word,
    src2: spirv::Word,
) -> Result<(), dr::Error> {
    let f64_type = map.get_or_add_scalar(builder, ast::ScalarType::F64);
    let f32_type = map.get_or_add_scalar(builder, ast::ScalarType::F32);
    let a = builder.f_convert(f64_type, None, src1)?;
    let b = builder.f_convert(f64_type, None, src2)?;
    let wide = match op {
        RoundedFloatOp::Add => builder.f_add(f64_type, None, a, b)?,
        RoundedFloatOp::Sub => builder.f_sub(f64_type, None, a, b)?,
        RoundedFloatOp::Mul => builder.f_mul(f64_type, None, a, b)?,
        RoundedFloatOp::Mad(addend) => {
            let c = builder.f_convert(f64_type, None, addend)?;
            builder.ext_inst(
                f64_type,
                None,
                opencl,
                spirv::CLOp::fma as spirv::Word,
                [
                    dr::Operand::IdRef(a),
                    dr::Operand::IdRef(b),
                    dr::Operand::IdRef(c),
                ]
                .iter()
                .cloned(),
            )?
        }
    };
    // The conversion is the only instruction the backend rounds as asked, and
    // the decoration has to sit on its own result.
    let converted = builder.id();
    builder.f_convert(f32_type, Some(converted), wide)?;
    emit_rounding_decoration(builder, converted, rounding);
    builder.copy_object(f32_type, Some(dst), converted)?;
    Ok(())
}

fn emit_saturating_decoration(builder: &mut dr::Builder, dst: u32, saturate: bool) {
    if saturate {
        builder.decorate(dst, spirv::Decoration::SaturatedConversion, iter::empty());
    }
}

// SPIR-V only allows the FPRoundingMode decoration on a conversion, so it goes
// on the temporary the conversion defines: a PTX register can be written by
// several instructions, so the register itself is not always defined by the
// conversion.
fn emit_rounding_decoration(
    builder: &mut dr::Builder,
    dst: spirv::Word,
    rounding: Option<ast::RoundingMode>,
) {
    if let Some(rounding) = rounding {
        builder.decorate(
            dst,
            spirv::Decoration::FPRoundingMode,
            [rounding.to_spirv()].iter().cloned(),
        );
    }
}

impl ast::RoundingMode {
    fn to_spirv(self) -> rspirv::dr::Operand {
        let mode = match self {
            ast::RoundingMode::NearestEven => spirv::FPRoundingMode::RTE,
            ast::RoundingMode::Zero => spirv::FPRoundingMode::RTZ,
            ast::RoundingMode::PositiveInf => spirv::FPRoundingMode::RTP,
            ast::RoundingMode::NegativeInf => spirv::FPRoundingMode::RTN,
        };
        rspirv::dr::Operand::FPRoundingMode(mode)
    }
}

fn emit_setp(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    setp: &ast::SetpData,
    arg: &ast::Arg4Setp<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let result_type = map.get_or_add(builder, SpirvType::Base(SpirvScalarKey::Pred));
    let result_id = Some(arg.dst1);
    let operand_1 = arg.src1;
    let operand_2 = arg.src2;
    match (setp.cmp_op, setp.typ.kind()) {
        (ast::SetpCompareOp::Eq, ast::ScalarKind::Signed)
        | (ast::SetpCompareOp::Eq, ast::ScalarKind::Unsigned)
        | (ast::SetpCompareOp::Eq, ast::ScalarKind::Bit) => {
            builder.i_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Eq, ast::ScalarKind::Float) => {
            builder.f_ord_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NotEq, ast::ScalarKind::Signed)
        | (ast::SetpCompareOp::NotEq, ast::ScalarKind::Unsigned)
        | (ast::SetpCompareOp::NotEq, ast::ScalarKind::Bit) => {
            builder.i_not_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NotEq, ast::ScalarKind::Float) => {
            builder.f_ord_not_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Less, ast::ScalarKind::Unsigned)
        | (ast::SetpCompareOp::Less, ast::ScalarKind::Bit) => {
            builder.u_less_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Less, ast::ScalarKind::Signed) => {
            builder.s_less_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Less, ast::ScalarKind::Float) => {
            builder.f_ord_less_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::LessOrEq, ast::ScalarKind::Unsigned)
        | (ast::SetpCompareOp::LessOrEq, ast::ScalarKind::Bit) => {
            builder.u_less_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::LessOrEq, ast::ScalarKind::Signed) => {
            builder.s_less_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::LessOrEq, ast::ScalarKind::Float) => {
            builder.f_ord_less_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Greater, ast::ScalarKind::Unsigned)
        | (ast::SetpCompareOp::Greater, ast::ScalarKind::Bit) => {
            builder.u_greater_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Greater, ast::ScalarKind::Signed) => {
            builder.s_greater_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::Greater, ast::ScalarKind::Float) => {
            builder.f_ord_greater_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::GreaterOrEq, ast::ScalarKind::Unsigned)
        | (ast::SetpCompareOp::GreaterOrEq, ast::ScalarKind::Bit) => {
            builder.u_greater_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::GreaterOrEq, ast::ScalarKind::Signed) => {
            builder.s_greater_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::GreaterOrEq, ast::ScalarKind::Float) => {
            builder.f_ord_greater_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NanEq, _) => {
            builder.f_unord_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NanNotEq, _) => {
            builder.f_unord_not_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NanLess, _) => {
            builder.f_unord_less_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NanLessOrEq, _) => {
            builder.f_unord_less_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NanGreater, _) => {
            builder.f_unord_greater_than(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::NanGreaterOrEq, _) => {
            builder.f_unord_greater_than_equal(result_type, result_id, operand_1, operand_2)
        }
        (ast::SetpCompareOp::IsAnyNan, _) => {
            let temp1 = builder.is_nan(result_type, None, operand_1)?;
            let temp2 = builder.is_nan(result_type, None, operand_2)?;
            builder.logical_or(result_type, result_id, temp1, temp2)
        }
        (ast::SetpCompareOp::IsNotNan, _) => {
            let temp1 = builder.is_nan(result_type, None, operand_1)?;
            let temp2 = builder.is_nan(result_type, None, operand_2)?;
            let any_nan = builder.logical_or(result_type, None, temp1, temp2)?;
            logical_not(builder, result_type, result_id, any_nan)
        }
        _ => todo!(),
    }?;
    Ok(())
}

// IGC carries two copies of the SPIR-V translator and compiles L0 SPIR-V with
// the old one (intel/intel-graphics-compiler#148), which needs this spelled as
// a select.
fn logical_not(
    builder: &mut dr::Builder,
    result_type: spirv::Word,
    result_id: Option<spirv::Word>,
    operand: spirv::Word,
) -> Result<spirv::Word, dr::Error> {
    let const_true = builder.constant_true(result_type, None);
    let const_false = builder.constant_false(result_type, None);
    builder.select(result_type, result_id, operand, const_false, const_true)
}

fn emit_mul_sint(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::MulSInt,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let instruction_type = ast::ScalarType::from(desc.typ);
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    match desc.control {
        ast::MulIntControl::Low => {
            builder.i_mul(inst_type, Some(arg.dst), arg.src1, arg.src2)?;
        }
        ast::MulIntControl::High => {
            builder.ext_inst(
                inst_type,
                Some(arg.dst),
                opencl,
                spirv::CLOp::s_mul_hi as spirv::Word,
                [dr::Operand::IdRef(arg.src1), dr::Operand::IdRef(arg.src2)]
                    .iter()
                    .cloned(),
            )?;
        }
        ast::MulIntControl::Wide => {
            let instr_width = instruction_type.size_of();
            let instr_kind = instruction_type.kind();
            let dst_type = ast::ScalarType::from_parts(instr_width * 2, instr_kind);
            let dst_type_id = map.get_or_add_scalar(builder, dst_type);
            // PTX mul.wide.{s,u}32 computes the full 64-bit product. IGC does not
            // provide __spirv_{S,U}MulExtended intrinsics, so widen the operands
            // to 64 bits (sign/zero extend) and multiply in 64-bit.
            let src1_wide = builder.s_convert(dst_type_id, None, arg.src1)?;
            let src2_wide = builder.s_convert(dst_type_id, None, arg.src2)?;
            builder.i_mul(dst_type_id, Some(arg.dst), src1_wide, src2_wide)?;
        }
    }
    Ok(())
}

fn emit_mul_uint(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    desc: &ast::MulUInt,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let instruction_type = ast::ScalarType::from(desc.typ);
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(desc.typ)));
    match desc.control {
        ast::MulIntControl::Low => {
            builder.i_mul(inst_type, Some(arg.dst), arg.src1, arg.src2)?;
        }
        ast::MulIntControl::High => {
            builder.ext_inst(
                inst_type,
                Some(arg.dst),
                opencl,
                spirv::CLOp::u_mul_hi as spirv::Word,
                [dr::Operand::IdRef(arg.src1), dr::Operand::IdRef(arg.src2)]
                    .iter()
                    .cloned(),
            )?;
        }
        ast::MulIntControl::Wide => {
            let instr_width = instruction_type.size_of();
            let instr_kind = instruction_type.kind();
            let dst_type = ast::ScalarType::from_parts(instr_width * 2, instr_kind);
            let dst_type_id = map.get_or_add_scalar(builder, dst_type);
            // Zero-extend to 64-bit and multiply: full 64-bit unsigned product.
            let src1_wide = builder.u_convert(dst_type_id, None, arg.src1)?;
            let src2_wide = builder.u_convert(dst_type_id, None, arg.src2)?;
            builder.i_mul(dst_type_id, Some(arg.dst), src1_wide, src2_wide)?;
        }
    }
    Ok(())
}

// Surprisingly, structs can't be bitcast, so we route everything through a vector
fn struct2_bitcast_to_wide(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    base_type_key: SpirvScalarKey,
    instruction_type: spirv::Word,
    dst: spirv::Word,
    dst_type_id: spirv::Word,
    src: spirv::Word,
) -> Result<(), dr::Error> {
    let low_bits = builder.composite_extract(instruction_type, None, src, [0].iter().copied())?;
    let high_bits = builder.composite_extract(instruction_type, None, src, [1].iter().copied())?;
    let vector_type = map.get_or_add(builder, SpirvType::Vector(base_type_key, 2));
    let vector =
        builder.composite_construct(vector_type, None, [low_bits, high_bits].iter().copied())?;
    builder.bitcast(dst_type_id, Some(dst), vector)?;
    Ok(())
}

/// Emits a device-scope memory fence: OpMemoryBarrier with acquire-release
/// ordering over workgroup and cross-workgroup memory.
///
/// The scope is always Device, even for a workgroup-scope `membar.cta`, because
/// this backend does not compile `ld.volatile` into an uncached read, so the
/// fence is the only thing that makes another workgroup's store visible to a
/// workgroup polling for it. Two measurements put it there, both on CUB's
/// device-wide scan (which polls a tile descriptor until its predecessor
/// publishes an inclusive prefix):
///
/// - `membar.cta` at PTX scope made torch.cumsum over 100000 int64 wrong in 5
///   runs of 12; the window never saw the write, walked off the tile-state
///   array and faulted 2 MiB below it. Device scope: 12 of 12 exact.
/// - CUB stops issuing that fence at all once gridDim.x reaches 500
///   (`NANOSLEEP` replaces `MEMBAR.SC.CTA` in both poll loops), which is sound
///   on NVIDIA because its poll is `LD.E.STRONG.GPU` and a PTX volatile load
///   there is uncached. Here the load is cached, so from exactly 500 tiles the
///   poll spun for ever: 448000 int64 (500 tiles of 896) timed out while
///   447104 (499 tiles) was exact, and the same lifted text at 500 tiles was
///   exact on an RTX 2070. A volatile load therefore carries this fence too.
///
/// A device fence is a superset of what a workgroup-scope barrier asks for, so
/// this synchronises more than the program requires rather than changing what
/// it means.
fn emit_device_fence(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
) -> Result<(), TranslateError> {
    let scope_id = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(ast::MemScope::Gpu.to_spirv() as u32),
    )?;
    let semantics_id = map.get_or_add_constant(
        builder,
        &ast::Type::Scalar(ast::ScalarType::U32),
        &vec_repr(
            spirv::MemorySemantics::ACQUIRE_RELEASE
                | spirv::MemorySemantics::WORKGROUP_MEMORY
                | spirv::MemorySemantics::CROSS_WORKGROUP_MEMORY,
        ),
    )?;
    builder.memory_barrier(scope_id, semantics_id)?;
    Ok(())
}

fn emit_abs(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    d: &ast::AbsDetails,
    arg: &ast::Arg2<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let scalar_t = ast::ScalarType::from(d.typ);
    let result_type = map.get_or_add(builder, SpirvType::from(scalar_t));
    let cl_abs = if scalar_t.kind() == ast::ScalarKind::Signed {
        spirv::CLOp::s_abs
    } else {
        spirv::CLOp::fabs
    };
    builder.ext_inst(
        result_type,
        Some(arg.dst),
        opencl,
        cl_abs as spirv::Word,
        [dr::Operand::IdRef(arg.src)].iter().cloned(),
    )?;
    Ok(())
}

fn emit_add_int(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    typ: ast::ScalarType,
    saturate: bool,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(typ)));
    if saturate {
        // The saturating add is the OpenCL one: it clamps to the range of the
        // type instead of wrapping, for every width.
        let op = if typ.kind() == ast::ScalarKind::Signed {
            spirv::CLOp::s_add_sat
        } else {
            spirv::CLOp::u_add_sat
        };
        builder.ext_inst(
            inst_type,
            Some(arg.dst),
            opencl,
            op as spirv::Word,
            [dr::Operand::IdRef(arg.src1), dr::Operand::IdRef(arg.src2)]
                .iter()
                .cloned(),
        )?;
        return Ok(());
    }
    builder.i_add(inst_type, Some(arg.dst), arg.src1, arg.src2)?;
    Ok(())
}

fn emit_sub_int(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    opencl: spirv::Word,
    typ: ast::ScalarType,
    saturate: bool,
    arg: &ast::Arg3<ExpandedArgParams>,
) -> Result<(), dr::Error> {
    let inst_type = map.get_or_add(builder, SpirvType::from(ast::ScalarType::from(typ)));
    if saturate {
        let op = if typ.kind() == ast::ScalarKind::Signed {
            spirv::CLOp::s_sub_sat
        } else {
            spirv::CLOp::u_sub_sat
        };
        builder.ext_inst(
            inst_type,
            Some(arg.dst),
            opencl,
            op as spirv::Word,
            [dr::Operand::IdRef(arg.src1), dr::Operand::IdRef(arg.src2)]
                .iter()
                .cloned(),
        )?;
        return Ok(());
    }
    builder.i_sub(inst_type, Some(arg.dst), arg.src1, arg.src2)?;
    Ok(())
}

fn emit_implicit_conversion(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    cv: &ImplicitConversion,
) -> Result<(), TranslateError> {
    let from_parts = cv.from_type.to_parts();
    let to_parts = cv.to_type.to_parts();
    match (from_parts.kind, to_parts.kind, &cv.kind) {
        (_, _, &ConversionKind::BitToPtr) => {
            let dst_type = map.get_or_add(
                builder,
                SpirvType::pointer_to(cv.to_type.clone(), cv.to_space.to_spirv()),
            );
            // BitToPtr in 64-bit PTX converts a pointer-sized value (int or
            // pointer) to a pointer. OpConvertUToPtr requires an int operand, so
            // a source that is already a pointer (e.g. after byte-pointer
            // arithmetic) is bitcast across storage classes instead.
            if from_parts.kind == TypeKind::Scalar {
                // PTX treats the address of a symbol as generic, and arithmetic
                // on it keeps it generic, so a number that names a shared or a
                // local address has to be cast into the space the access names.
                let space = cv.to_space.to_spirv();
                if space == spirv::StorageClass::Generic
                    || space == spirv::StorageClass::CrossWorkgroup
                {
                    builder.convert_u_to_ptr(dst_type, Some(cv.dst), cv.src)?;
                } else {
                    let generic_type = map.get_or_add(
                        builder,
                        SpirvType::pointer_to(cv.to_type.clone(), spirv::StorageClass::Generic),
                    );
                    let generic = builder.convert_u_to_ptr(generic_type, None, cv.src)?;
                    builder.generic_cast_to_ptr(dst_type, Some(cv.dst), generic)?;
                }
            } else {
                builder.bitcast(dst_type, Some(cv.dst), cv.src)?;
            }
        }
        (TypeKind::Scalar, TypeKind::Scalar, &ConversionKind::Default) => {
            if from_parts.width == to_parts.width {
                let dst_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
                // A packed pair of halves lives in a 32-bit register: only its
                // bit pattern matters, so it is reinterpreted, not converted.
                let float_kind = |kind: ast::ScalarKind| {
                    kind == ast::ScalarKind::Float || kind == ast::ScalarKind::Float2
                };
                if !float_kind(from_parts.scalar_kind) && !float_kind(to_parts.scalar_kind) {
                    builder.copy_object(dst_type, Some(cv.dst), cv.src)?;
                } else {
                    builder.bitcast(dst_type, Some(cv.dst), cv.src)?;
                }
            } else {
                if std::env::var_os("ZLUDA_TRACE_PTX").is_some() {
                    eprintln!(
                        "conversion: from width={}, to width={}",
                        from_parts.width, to_parts.width
                    );
                }
                let same_width_bit_type = map.get_or_add(
                    builder,
                    SpirvType::new(ast::Type::from_parts(TypeParts {
                        scalar_kind: ast::ScalarKind::Bit,
                        ..from_parts
                    })),
                );
                let same_width_bit_value = builder.bitcast(same_width_bit_type, None, cv.src)?;
                let wide_bit_type = ast::Type::from_parts(TypeParts {
                    scalar_kind: ast::ScalarKind::Bit,
                    ..to_parts
                });
                let wide_bit_type_spirv =
                    map.get_or_add(builder, SpirvType::new(wide_bit_type.clone()));
                if to_parts.scalar_kind == ast::ScalarKind::Unsigned
                    || to_parts.scalar_kind == ast::ScalarKind::Bit
                {
                    builder.u_convert(wide_bit_type_spirv, Some(cv.dst), same_width_bit_value)?;
                } else {
                    let conversion_fn = if from_parts.scalar_kind == ast::ScalarKind::Signed
                        && to_parts.scalar_kind == ast::ScalarKind::Signed
                    {
                        dr::Builder::s_convert
                    } else {
                        dr::Builder::u_convert
                    };
                    let wide_bit_value =
                        conversion_fn(builder, wide_bit_type_spirv, None, same_width_bit_value)?;
                    emit_implicit_conversion(
                        builder,
                        map,
                        &ImplicitConversion {
                            src: wide_bit_value,
                            dst: cv.dst,
                            from_type: wide_bit_type,
                            from_space: cv.from_space,
                            to_type: cv.to_type.clone(),
                            to_space: cv.to_space,
                            kind: ConversionKind::Default,
                        },
                    )?;
                }
            }
        }
        (TypeKind::Scalar, TypeKind::Scalar, &ConversionKind::Truncate) => {
            if std::env::var_os("ZLUDA_TRACE_PTX").is_some() {
                eprintln!(
                    "truncate: from width={} to width={}",
                    cv.from_type.size_of(),
                    cv.to_type.size_of()
                );
            }
            // The instruction wants a narrower value than the register holds, or
            // has written one. Reading one keeps the low bits (an 8-bit value
            // lives in the low byte of a 16-bit register in PTX); writing one
            // fills the upper byte from the narrow type's sign, so cvt.s8.s32 of
            // 0x80 leaves 0xff80. A narrow signed type only appears in the
            // writing direction, because the rule above does not apply to a
            // register declared signed.
            let result_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            if from_parts.scalar_kind == ast::ScalarKind::Signed {
                builder.s_convert(result_type, Some(cv.dst), cv.src)?;
            } else {
                builder.u_convert(result_type, Some(cv.dst), cv.src)?;
            }
        }
        (TypeKind::Scalar, TypeKind::Scalar, &ConversionKind::SignExtend) => {
            let result_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            builder.s_convert(result_type, Some(cv.dst), cv.src)?;
        }
        (TypeKind::Vector, TypeKind::Scalar, &ConversionKind::Default)
        | (TypeKind::Scalar, TypeKind::Array, &ConversionKind::Default)
        | (TypeKind::Array, TypeKind::Scalar, &ConversionKind::Default) => {
            let into_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            builder.bitcast(into_type, Some(cv.dst), cv.src)?;
        }
        (_, _, &ConversionKind::PtrToPtr) => {
            let result_type = map.get_or_add(
                builder,
                SpirvType::Pointer(
                    Box::new(SpirvType::new(cv.to_type.clone())),
                    cv.to_space.to_spirv(),
                ),
            );
            builder.bitcast(result_type, Some(cv.dst), cv.src)?;
        }
        (_, _, &ConversionKind::AddressOf) => {
            let dst_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            let address = generic_address(builder, map, cv.src, &cv.from_type, cv.from_space)?;
            builder.convert_ptr_to_u(dst_type, Some(cv.dst), address)?;
        }
        // A 64-bit value and a pair of 32-bit values hold the same bits
        // (`mov.b64 {%r0,%r1}, %rd2;`). SPIR-V cannot bitcast between a
        // scalar and a vector, so the halves are moved explicitly.
        (TypeKind::Vector, TypeKind::Scalar, &ConversionKind::Default)
            if cv.from_type.size_of() == cv.to_type.size_of() =>
        {
            // The source is a pair of 32-bit values; join them into the
            // 64-bit value the destination expects.
            let elem_type = match cv.from_type {
                ast::Type::Vector(scalar_t, _) => scalar_t,
                _ => unreachable!(),
            };
            let vector_type = map.get_or_add(
                builder,
                SpirvType::Vector(SpirvScalarKey::from(elem_type), 2),
            );
            let scalar_type = map.get_or_add(builder, SpirvType::new(ast::Type::Scalar(elem_type)));
            let wide_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            // The pieces are integers of the element's width: a packed pair
            // of half precision values is put together out of two 16-bit
            // pieces, a pair of 32-bit values out of two 32-bit ones.
            let elem_bits = elem_type.size_of() as u32 * 8;
            let int_elem_type = map.get_or_add_scalar(
                builder,
                match elem_bits {
                    16 => ast::ScalarType::U16,
                    _ => ast::ScalarType::U32,
                },
            );
            let low = builder.composite_extract(scalar_type, None, cv.src, [0].iter().copied())?;
            let high = builder.composite_extract(scalar_type, None, cv.src, [1].iter().copied())?;
            let low = if scalar_type == int_elem_type {
                low
            } else {
                builder.bitcast(int_elem_type, None, low)?
            };
            let high = if scalar_type == int_elem_type {
                high
            } else {
                builder.bitcast(int_elem_type, None, high)?
            };
            let low64 = builder.u_convert(wide_type, None, low)?;
            let high64 = builder.u_convert(wide_type, None, high)?;
            let shift = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U64),
                &vec_repr(elem_bits as u64),
            )?;
            let high_shifted = builder.shift_left_logical(wide_type, None, high64, shift)?;
            let joined = builder.bitwise_or(wide_type, None, low64, high_shifted)?;
            let result_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            builder.copy_object(result_type, Some(cv.dst), joined)?;
            let _ = vector_type;
        }
        (TypeKind::Scalar, TypeKind::Vector, &ConversionKind::Default)
            if cv.from_type.size_of() == cv.to_type.size_of() =>
        {
            // Split the 64-bit scalar into its two 32-bit halves.
            let elem_type = match cv.to_type {
                ast::Type::Vector(scalar_t, _) => scalar_t,
                _ => unreachable!(),
            };
            let scalar_type = map.get_or_add(builder, SpirvType::new(ast::Type::Scalar(elem_type)));
            let vector_type = map.get_or_add(
                builder,
                SpirvType::Vector(SpirvScalarKey::from(elem_type), 2),
            );
            // The source is a wider value; work in its width and cut the halves
            // out of it, as integers of the element's width.
            let wide_type = map.get_or_add(builder, SpirvType::new(cv.from_type.clone()));
            let elem_bits = match cv.to_type {
                ast::Type::Vector(scalar_t, _) => scalar_t.size_of() as u32 * 8,
                _ => unreachable!(),
            };
            let int_elem_type = map.get_or_add_scalar(
                builder,
                match elem_bits {
                    16 => ast::ScalarType::U16,
                    _ => ast::ScalarType::U32,
                },
            );
            let mask = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U64),
                &vec_repr((1u64 << elem_bits) - 1),
            )?;
            let shift = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U64),
                &vec_repr(elem_bits as u64),
            )?;
            let low64 = builder.bitwise_and(wide_type, None, cv.src, mask)?;
            let high64 = builder.shift_right_logical(wide_type, None, cv.src, shift)?;
            let low = builder.u_convert(int_elem_type, None, low64)?;
            let high = builder.u_convert(int_elem_type, None, high64)?;
            let low = if scalar_type == int_elem_type {
                low
            } else {
                builder.bitcast(scalar_type, None, low)?
            };
            let high = if scalar_type == int_elem_type {
                high
            } else {
                builder.bitcast(scalar_type, None, high)?
            };
            let vector = builder.composite_construct(
                vector_type,
                None,
                [low, high].iter().copied(),
            )?;
            let result_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            builder.bitcast(result_type, Some(cv.dst), vector)?;
        }
        (TypeKind::Pointer, TypeKind::Scalar, &ConversionKind::Default) => {
            let result_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            let address = generic_address(builder, map, cv.src, &cv.from_type, cv.from_space)?;
            builder.convert_ptr_to_u(result_type, Some(cv.dst), address)?;
        }        (TypeKind::Scalar, TypeKind::Pointer, &ConversionKind::Default) => {
            let result_type = map.get_or_add(builder, SpirvType::new(cv.to_type.clone()));
            builder.convert_u_to_ptr(result_type, Some(cv.dst), cv.src)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// The address of a pointer as a number a generic operation can use.
///
/// A shared or local pointer converts to its offset inside that space, which is
/// not an address a generic load may use; PTX treats the address of a symbol as
/// generic, so the pointer is cast to the generic space first.
fn generic_address(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    pointer: spirv::Word,
    pointee: &ast::Type,
    space: ast::StateSpace,
) -> Result<spirv::Word, TranslateError> {
    let storage_class = space.to_spirv();
    if storage_class == spirv::StorageClass::Generic
        || storage_class == spirv::StorageClass::CrossWorkgroup
    {
        return Ok(pointer);
    }
    // The cast keeps the pointee: a pointer to a pointer becomes a generic
    // pointer to the same thing, not a generic pointer to a pointer.
    let pointee_type = match SpirvType::new(pointee.clone()) {
        SpirvType::Pointer(inner, _) => *inner,
        other => other,
    };
    // An array is addressed by the pointer to its first element, and a shared
    // window substituted for an extern shared symbol is already a byte pointer,
    // so the operand of a cast on an array's address is not always
    // "pointer to array". OpPtrCastToGeneric requires its operand and its result
    // to point to the same type, so the address is reduced to a byte pointer
    // first, which is what both cases mean. The bitcast does not change the
    // address.
    let (pointee_type, pointer) = match pointee_type {
        SpirvType::Array(..) => {
            let byte_type = SpirvType::Base(SpirvScalarKey::B8);
            let byte_pointer_type =
                map.get_or_add(builder, SpirvType::Pointer(Box::new(byte_type.clone()), storage_class));
            (byte_type, builder.bitcast(byte_pointer_type, None, pointer)?)
        }
        other => (other, pointer),
    };
    let generic_type = map.get_or_add(
        builder,
        SpirvType::Pointer(Box::new(pointee_type), spirv::StorageClass::Generic),
    );
    Ok(builder.ptr_cast_to_generic(generic_type, None, pointer)?)
}

fn emit_load_var(
    builder: &mut dr::Builder,
    map: &mut TypeWordMap,
    details: &LoadVarDetails,
) -> Result<(), TranslateError> {
    let result_type = map.get_or_add(builder, SpirvType::new(details.typ.clone()));
    match details.member_index {
        Some((index, Some(width))) => {
            let vector_type = match details.typ {
                ast::Type::Scalar(scalar_t) => ast::Type::Vector(scalar_t, width),
                _ => return Err(error_mismatched_type()),
            };
            let vector_type_spirv = map.get_or_add(builder, SpirvType::new(vector_type));
            let vector_temp = builder.load(
                vector_type_spirv,
                None,
                details.arg.src,
                None,
                iter::empty(),
            )?;
            builder.composite_extract(
                result_type,
                Some(details.arg.dst),
                vector_temp,
                [index as u32].iter().copied(),
            )?;
        }
        Some((index, None)) => {
            let result_ptr_type = map.get_or_add(
                builder,
                SpirvType::pointer_to(details.typ.clone(), spirv::StorageClass::Function),
            );
            let index_spirv = map.get_or_add_constant(
                builder,
                &ast::Type::Scalar(ast::ScalarType::U32),
                &vec_repr(index as u32),
            )?;
            let src = builder.in_bounds_access_chain(
                result_ptr_type,
                None,
                details.arg.src,
                [index_spirv].iter().copied(),
            )?;
            builder.load(result_type, Some(details.arg.dst), src, None, iter::empty())?;
        }
        None => {
            builder.load(
                result_type,
                Some(details.arg.dst),
                details.arg.src,
                None,
                iter::empty(),
            )?;
        }
    };
    Ok(())
}

fn normalize_identifiers<'input, 'b>(
    id_defs: &mut FnStringIdResolver<'input, 'b>,
    fn_defs: &GlobalFnDeclResolver<'input, 'b>,
    func: Vec<ast::Statement<ast::ParsedArgParams<'input>>>,
) -> Result<Vec<NormalizedStatement>, TranslateError> {
    for s in func.iter() {
        match s {
            ast::Statement::Label(id) | ast::Statement::BranchTargets(id, _) => {
                id_defs.add_def(*id, None, false);
            }
            _ => (),
        }
    }
    // The jump tables of the function: the label that names the table, and the
    // labels of its targets in order.
    let mut branch_tables: HashMap<&str, Vec<&str>> = HashMap::new();
    for s in func.iter() {
        if let ast::Statement::BranchTargets(name, targets) = s {
            branch_tables.insert(*name, targets.clone());
        }
    }
    let mut result = Vec::new();
    for s in func {
        expand_map_variables(id_defs, fn_defs, &branch_tables, &mut result, s)?;
    }
    Ok(result)
}

fn expand_map_variables<'a, 'b>(
    id_defs: &mut FnStringIdResolver<'a, 'b>,
    fn_defs: &GlobalFnDeclResolver<'a, 'b>,
    branch_tables: &HashMap<&str, Vec<&str>>,
    result: &mut Vec<NormalizedStatement>,
    s: ast::Statement<ast::ParsedArgParams<'a>>,
) -> Result<(), TranslateError> {
    match s {
        ast::Statement::Block(block) => {
            // A block that only declares variables (".reg .f16 low, high;")
            // does not open a new scope: the registers belong to the
            // enclosing block, which is where they are used.
            let only_variables = block
                .iter()
                .all(|s| matches!(s, ast::Statement::Variable(_)));
            if !only_variables {
                id_defs.start_block();
            }
            for s in block {
                expand_map_variables(id_defs, fn_defs, branch_tables, result, s)?;
            }
            if !only_variables {
                id_defs.end_block();
            }
        }
        ast::Statement::Label(name) => result.push(Statement::Label(id_defs.get_id(name)?)),
        // A jump table is not code: its labels are used by the branch.
        ast::Statement::BranchTargets(..) => {}
        ast::Statement::Instruction(p, ast::Instruction::BrxIdx(index, table, _)) => {
            let table_name = match &table {
                Some(ast::Operand::Reg(name)) => *name,
                _ => return Err(error_unreachable()),
            };
            let targets = branch_tables
                .get(table_name)
                .ok_or_else(|| TranslateError::UnknownSymbolName(table_name.to_string()))?;
            let mut target_ids = Vec::with_capacity(targets.len());
            for target in targets.iter() {
                target_ids.push(id_defs.get_id(target)?);
            }
            let index = index.map_variable(&mut |id| id_defs.get_id(id))?;
            result.push(Statement::Instruction((
                p.map(|p| p.map_variable(&mut |id| id_defs.get_id(id)))
                    .transpose()?,
                ast::Instruction::BrxIdx(index, None, target_ids),
            )))
        }
        ast::Statement::Instruction(p, i) => result.push(Statement::Instruction((
            p.map(|p| p.map_variable(&mut |id| id_defs.get_id(id)))
                .transpose()?,
            i.map_variable(&mut |id| id_defs.get_id(id))?,
        ))),
        ast::Statement::Variable(var) => {
            let var_type = var.var.v_type.clone();
            match var.count {
                Some(count) => {
                    for new_id in
                        id_defs.add_defs(var.var.name, count, var_type, var.var.state_space, true)
                    {
                        result.push(Statement::Variable(ast::Variable {
                            align: var.var.align,
                            v_type: var.var.v_type.clone(),
                            state_space: var.var.state_space,
                            name: new_id,
                            array_init: var.var.array_init.clone(),
                        }))
                    }
                }
                None => {
                    let new_id =
                        id_defs.add_def(var.var.name, Some((var_type, var.var.state_space)), true);
                    result.push(Statement::Variable(ast::Variable {
                        align: var.var.align,
                        v_type: var.var.v_type.clone(),
                        state_space: var.var.state_space,
                        name: new_id,
                        array_init: var.var.array_init,
                    }));
                }
            }
            // Further registers of the same declaration: ".reg .f16 a, b;"
            for extra in var.extra {
                let extra_type = extra.v_type.clone();
                let new_id =
                    id_defs.add_def(extra.name, Some((extra_type, extra.state_space)), true);
                result.push(Statement::Variable(ast::Variable {
                    align: extra.align,
                    v_type: extra.v_type,
                    state_space: extra.state_space,
                    name: new_id,
                    array_init: extra.array_init,
                }));
            }
        }
    };
    Ok(())
}

/*
    Our goal here is to transform
        .visible .entry foobar(.param .u64 input) {
            .reg .b64 in_addr;
            .reg .b64 in_addr2;
            ld.param.u64 in_addr, [input];
            cvta.to.global.u64  in_addr2, in_addr;
        }
    into:
        .visible .entry foobar(.param .u8 input[]) {
            .reg .u8 in_addr[];
            .reg .u8 in_addr2[];
            ld.param.u8[] in_addr, [input];
            mov.u8[] in_addr2, in_addr;
        }
    or:
        .visible .entry foobar(.reg .u8 input[]) {
            .reg .u8 in_addr[];
            .reg .u8 in_addr2[];
            mov.u8[] in_addr, input;
            mov.u8[] in_addr2, in_addr;
        }
    or:
        .visible .entry foobar(.param ptr<u8, global> input) {
            .reg ptr<u8, global> in_addr;
            .reg ptr<u8, global> in_addr2;
            ld.param.ptr<u8, global> in_addr, [input];
            mov.ptr<u8, global> in_addr2, in_addr;
        }
*/
// TODO: detect more patterns (mov, call via reg, call via param)
// TODO: don't convert to ptr if the register is not ultimately used for ld/st
// TODO: once insert_mem_ssa_statements is moved to later, move this pass after
//       argument expansion
// TODO: propagate out of calls and into calls
fn convert_to_stateful_memory_access<'a, 'input>(
    func_args: Rc<RefCell<ast::MethodDeclaration<'input, spirv::Word>>>,
    func_body: Vec<TypedStatement>,
    id_defs: &mut NumericIdResolver<'a>,
) -> Result<
    (
        Rc<RefCell<ast::MethodDeclaration<'input, spirv::Word>>>,
        Vec<TypedStatement>,
    ),
    TranslateError,
> {
    let mut method_decl = func_args.borrow_mut();
    if !method_decl.name.is_kernel() {
        drop(method_decl);
        return Ok((func_args, func_body));
    }
    if Rc::strong_count(&func_args) != 1 {
        return Err(error_unreachable());
    }
    let func_args_64bit = (*method_decl)
        .input_arguments
        .iter()
        .filter_map(|arg| match arg.v_type {
            ast::Type::Scalar(ast::ScalarType::U64)
            | ast::Type::Scalar(ast::ScalarType::B64)
            | ast::Type::Scalar(ast::ScalarType::S64) => Some(arg.name),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut stateful_markers = Vec::new();
    let mut stateful_init_reg = MultiHashMap::new();
    for statement in func_body.iter() {
        match statement {
            Statement::Instruction(ast::Instruction::Cvta(
                ast::CvtaDetails {
                    to: ast::StateSpace::Global,
                    size: ast::CvtaSize::U64,
                    from: ast::StateSpace::Generic,
                },
                arg,
            )) => {
                if let (TypedOperand::Reg(dst), Some(src)) =
                    (arg.dst, arg.src.upcast().underlying_register())
                {
                    if is_64_bit_integer(id_defs, *src) && is_64_bit_integer(id_defs, dst) {
                        stateful_markers.push((dst, *src));
                    }
                }
            }
            Statement::Instruction(ast::Instruction::Ld(
                ast::LdDetails {
                    state_space: ast::StateSpace::Param,
                    typ: ast::Type::Scalar(ast::ScalarType::U64),
                    ..
                },
                arg,
            ))
            | Statement::Instruction(ast::Instruction::Ld(
                ast::LdDetails {
                    state_space: ast::StateSpace::Param,
                    typ: ast::Type::Scalar(ast::ScalarType::S64),
                    ..
                },
                arg,
            ))
            | Statement::Instruction(ast::Instruction::Ld(
                ast::LdDetails {
                    state_space: ast::StateSpace::Param,
                    typ: ast::Type::Scalar(ast::ScalarType::B64),
                    ..
                },
                arg,
            )) => {
                if let (TypedOperand::Reg(dst), Some(src)) =
                    (&arg.dst, arg.src.upcast().underlying_register())
                {
                    if func_args_64bit.contains(src) {
                        multi_hash_map_append(&mut stateful_init_reg, *dst, *src);
                    }
                }
            }
            _ => {}

        }
    }
    if stateful_markers.len() == 0 {
        drop(method_decl);
        return Ok((func_args, func_body));
    }
    // Registers whose base is a .shared symbol must be Shared (Workgroup)
    // pointers, not Global (CrossWorkgroup), or the shared accesses they feed
    // roundtrip through the wrong address space.
    let mut shared_ptr_dsts = HashSet::new();
    for statement in func_body.iter() {
        if let Statement::Instruction(ast::Instruction::Mov(_, arg)) = statement {
            if let (TypedOperand::Reg(dst), Some(src)) =
                (arg.dst, arg.src.upcast().underlying_register())
            {
                if let Ok((_, sp, _)) = id_defs.get_typed(*src) {
                    if sp == ast::StateSpace::Shared {
                        shared_ptr_dsts.insert(dst);
                    }
                }
            }
        }
    }
    let mut func_args_ptr = HashSet::new();
    let mut regs_ptr_current = HashSet::new();
    for (dst, src) in stateful_markers {
        if let Some(func_args) = stateful_init_reg.get(&src) {
            for a in func_args {
                func_args_ptr.insert(*a);
                regs_ptr_current.insert(src);
                regs_ptr_current.insert(dst);
            }
        }
    }
    // BTreeSet here to have a stable order of iteration,
    // unfortunately our tests rely on it
    let mut regs_ptr_seen = BTreeSet::new();
    while regs_ptr_current.len() > 0 {
        let mut regs_ptr_new = HashSet::new();
        for statement in func_body.iter() {
            match statement {
                Statement::Instruction(ast::Instruction::Add(
                    ast::ArithDetails::Unsigned(ast::ScalarType::U64),
                    arg,
                ))
                | Statement::Instruction(ast::Instruction::Add(
                    ast::ArithDetails::Signed(ast::ArithSInt {
                        typ: ast::ScalarType::S64,
                        saturate: false,
                    }),
                    arg,
                ))
                | Statement::Instruction(ast::Instruction::Sub(
                    ast::ArithDetails::Unsigned(ast::ScalarType::U64),
                    arg,
                ))
                | Statement::Instruction(ast::Instruction::Sub(
                    ast::ArithDetails::Signed(ast::ArithSInt {
                        typ: ast::ScalarType::S64,
                        saturate: false,
                    }),
                    arg,
                )) => {
                    // TODO: don't mark result of double pointer sub or double
                    // pointer add as ptr result
                    if let (TypedOperand::Reg(dst), Some(src1)) =
                        (arg.dst, arg.src1.upcast().underlying_register())
                    {
                        if regs_ptr_current.contains(src1) && !regs_ptr_seen.contains(src1) {
                            regs_ptr_new.insert(dst);
                        }
                    } else if let (TypedOperand::Reg(dst), Some(src2)) =
                        (arg.dst, arg.src2.upcast().underlying_register())
                    {
                        if regs_ptr_current.contains(src2) && !regs_ptr_seen.contains(src2) {
                            regs_ptr_new.insert(dst);
                        }
                    }
                }
                _ => {}
            }
        }
        for id in regs_ptr_current {
            regs_ptr_seen.insert(id);
        }
        regs_ptr_current = regs_ptr_new;
    }
    drop(regs_ptr_current);
    let mut remapped_ids = HashMap::new();
    let mut result = Vec::with_capacity(regs_ptr_seen.len() + func_body.len());
    for reg in regs_ptr_seen {
        let space = if shared_ptr_dsts.contains(&reg) {
            ast::StateSpace::Shared
        } else {
            ast::StateSpace::Global
        };
        let new_id = id_defs.register_variable(
            ast::Type::Pointer(ast::ScalarType::U8, space),
            ast::StateSpace::Reg,
        );
        result.push(Statement::Variable(ast::Variable {
            align: None,
            name: new_id,
            array_init: Vec::new(),
            v_type: ast::Type::Pointer(ast::ScalarType::U8, space),
            state_space: ast::StateSpace::Reg,
        }));
        remapped_ids.insert(reg, new_id);
    }
    for arg in (*method_decl).input_arguments.iter_mut() {
        if !func_args_ptr.contains(&arg.name) {
            continue;
        }
        let new_id = id_defs.register_variable(
            ast::Type::Pointer(ast::ScalarType::U8, ast::StateSpace::Global),
            ast::StateSpace::Param,
        );
        let old_name = arg.name;
        arg.v_type = ast::Type::Pointer(ast::ScalarType::U8, ast::StateSpace::Global);
        arg.name = new_id;
        remapped_ids.insert(old_name, new_id);
    }
    for statement in func_body {
        match statement {
            l @ Statement::Label(_) => result.push(l),
            c @ Statement::Conditional(_) => result.push(c),
            // A constant needs no stateful memory access processing.
            c @ Statement::Constant(_) => result.push(c),
            Statement::Variable(var) => {
                if !remapped_ids.contains_key(&var.name) {
                    result.push(Statement::Variable(var));
                }
            }
            Statement::Instruction(ast::Instruction::Add(
                ast::ArithDetails::Unsigned(ast::ScalarType::U64),
                arg,
            ))
            | Statement::Instruction(ast::Instruction::Add(
                ast::ArithDetails::Signed(ast::ArithSInt {
                    typ: ast::ScalarType::S64,
                    saturate: false,
                }),
                arg,
            )) if is_add_ptr_direct(&remapped_ids, &arg) => {
                let (ptr, offset) = match arg.src1.upcast().underlying_register() {
                    Some(src1) if remapped_ids.contains_key(src1) => {
                        (remapped_ids.get(src1).unwrap(), arg.src2)
                    }
                    Some(src2) if remapped_ids.contains_key(src2) => {
                        (remapped_ids.get(src2).unwrap(), arg.src1)
                    }
                    _ => return Err(error_unreachable()),
                };
                let dst = arg.dst.upcast().unwrap_reg()?;
                result.push(Statement::PtrAccess(PtrAccess {
                    underlying_type: ast::Type::Scalar(ast::ScalarType::U8),
                    state_space: ast::StateSpace::Global,
                    dst: *remapped_ids.get(&dst).unwrap(),
                    ptr_src: *ptr,
                    offset_src: offset,
                }))
            }
            Statement::Instruction(ast::Instruction::Sub(
                ast::ArithDetails::Unsigned(ast::ScalarType::U64),
                arg,
            ))
            | Statement::Instruction(ast::Instruction::Sub(
                ast::ArithDetails::Signed(ast::ArithSInt {
                    typ: ast::ScalarType::S64,
                    saturate: false,
                }),
                arg,
            )) if is_sub_ptr_direct(&remapped_ids, &arg) => {
                let (ptr, offset) = match arg.src1.upcast().underlying_register() {
                    Some(src1) => (remapped_ids.get(src1).unwrap(), arg.src2),
                    _ => return Err(error_unreachable()),
                };
                let offset_neg = id_defs.register_intermediate(Some((
                    ast::Type::Scalar(ast::ScalarType::S64),
                    ast::StateSpace::Reg,
                )));
                result.push(Statement::Instruction(ast::Instruction::Neg(
                    ast::NegDetails {
                        typ: ast::ScalarType::S64,
                        flush_to_zero: None,
                    },
                    ast::Arg2 {
                        src: offset,
                        dst: TypedOperand::Reg(offset_neg),
                    },
                )));
                let dst = arg.dst.upcast().unwrap_reg()?;
                result.push(Statement::PtrAccess(PtrAccess {
                    underlying_type: ast::Type::Scalar(ast::ScalarType::U8),
                    state_space: ast::StateSpace::Global,
                    dst: *remapped_ids.get(&dst).unwrap(),
                    ptr_src: *ptr,
                    offset_src: TypedOperand::Reg(offset_neg),
                }))
            }
            Statement::Instruction(inst) => {
                let mut post_statements = Vec::new();
                let new_statement =
                    inst.visit(&mut |arg_desc, expected_type: Option<(&ast::Type, _)>| {
                        convert_to_stateful_memory_access_postprocess(
                            id_defs,
                            &remapped_ids,
                            &mut result,
                            &mut post_statements,
                            arg_desc,
                            expected_type,
                        )
                    })?;
                result.push(new_statement);
                result.extend(post_statements);
            }
            Statement::Call(call) => {
                let mut post_statements = Vec::new();
                let new_statement =
                    call.visit(&mut |arg_desc, expected_type: Option<(&ast::Type, _)>| {
                        convert_to_stateful_memory_access_postprocess(
                            id_defs,
                            &remapped_ids,
                            &mut result,
                            &mut post_statements,
                            arg_desc,
                            expected_type,
                        )
                    })?;
                result.push(new_statement);
                result.extend(post_statements);
            }
            Statement::RepackVector(pack) => {
                let mut post_statements = Vec::new();
                let new_statement =
                    pack.visit(&mut |arg_desc, expected_type: Option<(&ast::Type, _)>| {
                        convert_to_stateful_memory_access_postprocess(
                            id_defs,
                            &remapped_ids,
                            &mut result,
                            &mut post_statements,
                            arg_desc,
                            expected_type,
                        )
                    })?;
                result.push(new_statement);
                result.extend(post_statements);
            }
            _ => return Err(error_unreachable()),
        }
    }
    drop(method_decl);
    Ok((func_args, result))
}

fn convert_to_stateful_memory_access_postprocess(
    id_defs: &mut NumericIdResolver,
    remapped_ids: &HashMap<spirv::Word, spirv::Word>,
    result: &mut Vec<TypedStatement>,
    post_statements: &mut Vec<TypedStatement>,
    arg_desc: ArgumentDescriptor<spirv::Word>,
    expected_type: Option<(&ast::Type, ast::StateSpace)>,
) -> Result<spirv::Word, TranslateError> {
    Ok(match remapped_ids.get(&arg_desc.op) {
        Some(new_id) => {
            let (new_operand_type, new_operand_space, _) = id_defs.get_typed(*new_id)?;
            if let Some((expected_type, expected_space)) = expected_type {
                let implicit_conversion = arg_desc
                    .non_default_implicit_conversion
                    .unwrap_or(default_implicit_conversion);
                if implicit_conversion(
                    (new_operand_space, &new_operand_type),
                    (expected_space, expected_type),
                )
                .is_ok()
                {
                    return Ok(*new_id);
                }
            }
            let (old_operand_type, old_operand_space, _) = id_defs.get_typed(arg_desc.op)?;
            let converting_id =
                id_defs.register_intermediate(Some((old_operand_type.clone(), old_operand_space)));
            let kind = if new_operand_space.is_compatible(ast::StateSpace::Reg) {
                ConversionKind::Default
            } else {
                ConversionKind::PtrToPtr
            };
            if arg_desc.is_dst {
                post_statements.push(Statement::Conversion(ImplicitConversion {
                    src: converting_id,
                    dst: *new_id,
                    from_type: old_operand_type,
                    from_space: old_operand_space,
                    to_type: new_operand_type,
                    to_space: new_operand_space,
                    kind,
                }));
                converting_id
            } else {
                result.push(Statement::Conversion(ImplicitConversion {
                    src: *new_id,
                    dst: converting_id,
                    from_type: new_operand_type,
                    from_space: new_operand_space,
                    to_type: old_operand_type,
                    to_space: old_operand_space,
                    kind,
                }));
                converting_id
            }
        }
        None => {
            arg_desc.op
        }
    })
}

fn is_add_ptr_direct(remapped_ids: &HashMap<u32, u32>, arg: &ast::Arg3<TypedArgParams>) -> bool {
    match arg.dst {
        TypedOperand::Imm(..) | TypedOperand::RegOffset(..) | TypedOperand::VecMember(..) => {
            return false
        }
        TypedOperand::Reg(dst) => {
            if !remapped_ids.contains_key(&dst) {
                return false;
            }
            if let Some(src1_reg) = arg.src1.upcast().underlying_register() {
                if remapped_ids.contains_key(src1_reg) {
                    // don't trigger optimization when adding two pointers
                    if let Some(src2_reg) = arg.src2.upcast().underlying_register() {
                        return !remapped_ids.contains_key(src2_reg);
                    }
                }
            }
            if let Some(src2_reg) = arg.src2.upcast().underlying_register() {
                remapped_ids.contains_key(src2_reg)
            } else {
                false
            }
        }
    }
}

fn is_sub_ptr_direct(remapped_ids: &HashMap<u32, u32>, arg: &ast::Arg3<TypedArgParams>) -> bool {
    match arg.dst {
        TypedOperand::Imm(..) | TypedOperand::RegOffset(..) | TypedOperand::VecMember(..) => {
            return false
        }
        TypedOperand::Reg(dst) => {
            if !remapped_ids.contains_key(&dst) {
                return false;
            }
            match arg.src1.upcast().underlying_register() {
                Some(src1_reg) => {
                    if remapped_ids.contains_key(src1_reg) {
                        // don't trigger optimization when subtracting two pointers
                        arg.src2
                            .upcast()
                            .underlying_register()
                            .map_or(true, |src2_reg| !remapped_ids.contains_key(src2_reg))
                    } else {
                        false
                    }
                }
                None => false,
            }
        }
    }
}

fn is_64_bit_integer(id_defs: &NumericIdResolver, id: spirv::Word) -> bool {
    match id_defs.get_typed(id) {
        Ok((ast::Type::Scalar(ast::ScalarType::U64), _, _))
        | Ok((ast::Type::Scalar(ast::ScalarType::S64), _, _))
        | Ok((ast::Type::Scalar(ast::ScalarType::B64), _, _)) => true,
        _ => false,
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
enum PtxSpecialRegister {
    Tid,
    Tid64,
    Ntid,
    Ntid64,
    Ctaid,
    Ctaid64,
    Nctaid,
    Nctaid64,
    // %laneid: the thread's index inside its warp.
    Laneid,
    // %warpid: the warp's index inside the CTA.
    Warpid,
    // %nwarpid: the number of warps in the CTA.
    Nwarpid,
    // %lanemask_*: a mask of the lanes of the warp relative to this one.
    // Not SPIR-V builtins; computed from the lane id in
    // fix_special_registers.
    LanemaskLt,
    LanemaskLe,
    LanemaskEq,
    LanemaskGe,
    LanemaskGt,
    // CUDA kernel parameter-buffer base registers (%envreg1..%envreg32).
    // Not SPIR-V builtins; lowered to zero constants in fix_special_registers.
    Envreg1,
    Envreg2,
}

impl PtxSpecialRegister {
    fn try_parse(s: &str) -> Option<Self> {
        match s {
            "%tid" => Some(Self::Tid),
            "%ntid" => Some(Self::Ntid),
            "%ctaid" => Some(Self::Ctaid),
            "%nctaid" => Some(Self::Nctaid),
            "%laneid" => Some(Self::Laneid),
            "%warpid" => Some(Self::Warpid),
            "%nwarpid" => Some(Self::Nwarpid),
            "%lanemask_lt" => Some(Self::LanemaskLt),
            "%lanemask_le" => Some(Self::LanemaskLe),
            "%lanemask_eq" => Some(Self::LanemaskEq),
            "%lanemask_ge" => Some(Self::LanemaskGe),
            "%lanemask_gt" => Some(Self::LanemaskGt),
            "%envreg1" => Some(Self::Envreg1),
            "%envreg2" => Some(Self::Envreg2),
            _ => None,
        }
    }

    fn get_type(self) -> ast::Type {
        match self {
            PtxSpecialRegister::Tid => ast::Type::Vector(ast::ScalarType::U32, 4),
            PtxSpecialRegister::Tid64 => ast::Type::Vector(ast::ScalarType::U64, 3),
            PtxSpecialRegister::Ntid => ast::Type::Vector(ast::ScalarType::U32, 4),
            PtxSpecialRegister::Ntid64 => ast::Type::Vector(ast::ScalarType::U64, 3),
            PtxSpecialRegister::Ctaid => ast::Type::Vector(ast::ScalarType::U32, 4),
            PtxSpecialRegister::Ctaid64 => ast::Type::Vector(ast::ScalarType::U64, 3),
            PtxSpecialRegister::Nctaid => ast::Type::Vector(ast::ScalarType::U32, 4),
            PtxSpecialRegister::Nctaid64 => ast::Type::Vector(ast::ScalarType::U64, 3),
            PtxSpecialRegister::Laneid
            | PtxSpecialRegister::Warpid
            | PtxSpecialRegister::Nwarpid
            | PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskLe
            | PtxSpecialRegister::LanemaskEq
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::LanemaskGt => ast::Type::Scalar(ast::ScalarType::U32),
            PtxSpecialRegister::Envreg1 | PtxSpecialRegister::Envreg2 => {
                ast::Type::Scalar(ast::ScalarType::U32)
            }
        }
    }

    fn get_builtin(self) -> spirv::BuiltIn {
        match self {
            PtxSpecialRegister::Tid | PtxSpecialRegister::Tid64 => {
                spirv::BuiltIn::LocalInvocationId
            }
            PtxSpecialRegister::Ntid | PtxSpecialRegister::Ntid64 => {
                spirv::BuiltIn::EnqueuedWorkgroupSize
            }
            PtxSpecialRegister::Ctaid | PtxSpecialRegister::Ctaid64 => spirv::BuiltIn::WorkgroupId,
            PtxSpecialRegister::Nctaid | PtxSpecialRegister::Nctaid64 => {
                spirv::BuiltIn::NumWorkgroups
            }
            // %laneid is the x component of the subgroup-local invocation id.
            PtxSpecialRegister::Laneid => spirv::BuiltIn::SubgroupLocalInvocationId,
            // %warpid / %nwarpid have no direct SPIR-V builtin; they are
            // derived from the subgroup id and the subgroup size.
            PtxSpecialRegister::Warpid | PtxSpecialRegister::Nwarpid => {
                spirv::BuiltIn::SubgroupId
            }
            // Computed from the lane id, not a builtin of its own.
            PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskLe
            | PtxSpecialRegister::LanemaskEq
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::LanemaskGt => unreachable!(),
            // CUDA-specific, no SPIR-V equivalent (never emitted as a builtin)
            PtxSpecialRegister::Envreg1 | PtxSpecialRegister::Envreg2 => unreachable!(),
        }
    }

    /// True for registers that map to a real SPIR-V BuiltIn variable.
    /// The envregs and the lane masks are lowered by hand and must stay out of
    /// the builtin/interface emission.
    fn is_spirv_builtin(self) -> bool {
        !matches!(
            self,
            PtxSpecialRegister::Envreg1
                | PtxSpecialRegister::Envreg2
                | PtxSpecialRegister::LanemaskLt
                | PtxSpecialRegister::LanemaskLe
                | PtxSpecialRegister::LanemaskEq
                | PtxSpecialRegister::LanemaskGe
                | PtxSpecialRegister::LanemaskGt
        )
    }

    fn normalized_sreg_and_type(self) -> Option<(PtxSpecialRegister, ast::ScalarType, u8)> {
        match self {
            PtxSpecialRegister::Tid => Some((PtxSpecialRegister::Tid64, ast::ScalarType::U64, 3)),
            PtxSpecialRegister::Ntid => Some((PtxSpecialRegister::Ntid64, ast::ScalarType::U64, 3)),
            PtxSpecialRegister::Ctaid => {
                Some((PtxSpecialRegister::Ctaid64, ast::ScalarType::U64, 3))
            }
            PtxSpecialRegister::Nctaid => {
                Some((PtxSpecialRegister::Nctaid64, ast::ScalarType::U64, 3))
            }
            PtxSpecialRegister::Tid64
            | PtxSpecialRegister::Ntid64
            | PtxSpecialRegister::Ctaid64
            | PtxSpecialRegister::Nctaid64 => None,
            // The warp-level registers are already 32-bit scalars; there is no
            // 64-bit counterpart, so they are used as-is.
            PtxSpecialRegister::Laneid
            | PtxSpecialRegister::Warpid
            | PtxSpecialRegister::Nwarpid
            | PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskLe
            | PtxSpecialRegister::LanemaskEq
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::LanemaskGt => None,
            // Envregs have no normalized 64-bit counterpart; they are lowered
            // to zero constants before SSA emission.
            PtxSpecialRegister::Envreg1 | PtxSpecialRegister::Envreg2 => None,
        }
    }
}

struct SpecialRegistersMap {
    reg_to_id: HashMap<PtxSpecialRegister, spirv::Word>,
    id_to_reg: HashMap<spirv::Word, PtxSpecialRegister>,
}

impl SpecialRegistersMap {
    fn new() -> Self {
        SpecialRegistersMap {
            reg_to_id: HashMap::new(),
            id_to_reg: HashMap::new(),
        }
    }

    fn builtins<'a>(&'a self) -> impl Iterator<Item = (PtxSpecialRegister, spirv::Word)> + 'a {
        self.reg_to_id.iter().filter_map(|(sreg, id)| {
            if sreg.is_spirv_builtin() && sreg.normalized_sreg_and_type().is_none() {
                Some((*sreg, *id))
            } else {
                None
            }
        })
    }

    fn interface(&self) -> Vec<spirv::Word> {
        self.reg_to_id
            .iter()
            .filter_map(|(sreg, id)| {
                if sreg.is_spirv_builtin() && sreg.normalized_sreg_and_type().is_none() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    fn get(&self, id: spirv::Word) -> Option<PtxSpecialRegister> {
        self.id_to_reg.get(&id).copied()
    }

    fn get_or_add(&mut self, current_id: &mut spirv::Word, reg: PtxSpecialRegister) -> spirv::Word {
        match self.reg_to_id.entry(reg) {
            hash_map::Entry::Occupied(e) => *e.get(),
            hash_map::Entry::Vacant(e) => {
                let numeric_id = *current_id;
                *current_id += 1;
                e.insert(numeric_id);
                self.id_to_reg.insert(numeric_id, reg);
                numeric_id
            }
        }
    }
}

struct FnSigMapper<'input> {
    // true - stays as return argument
    // false - is moved to input argument
    return_param_args: Vec<bool>,
    func_decl: Rc<RefCell<ast::MethodDeclaration<'input, spirv::Word>>>,
}

impl<'input> FnSigMapper<'input> {
    fn remap_to_spirv_repr(mut method: ast::MethodDeclaration<'input, spirv::Word>) -> Self {
        let return_param_args = method
            .return_arguments
            .iter()
            .map(|a| a.state_space != ast::StateSpace::Param)
            .collect::<Vec<_>>();
        let mut new_return_arguments = Vec::new();
        for arg in method.return_arguments.into_iter() {
            if arg.state_space == ast::StateSpace::Param {
                method.input_arguments.push(arg);
            } else {
                new_return_arguments.push(arg);
            }
        }
        method.return_arguments = new_return_arguments;
        FnSigMapper {
            return_param_args,
            func_decl: Rc::new(RefCell::new(method)),
        }
    }

    fn resolve_in_spirv_repr(
        &self,
        call_inst: ast::CallInst<NormalizedArgParams>,
    ) -> Result<ResolvedCall<NormalizedArgParams>, TranslateError> {
        let func_decl = (*self.func_decl).borrow();
        let mut return_arguments = Vec::new();
        let mut input_arguments = call_inst
            .param_list
            .into_iter()
            .zip(func_decl.input_arguments.iter())
            .map(|(id, var)| (id, var.v_type.clone(), var.state_space))
            .collect::<Vec<_>>();
        let mut func_decl_return_iter = func_decl.return_arguments.iter();
        let mut func_decl_input_iter = func_decl.input_arguments[input_arguments.len()..].iter();
        let call_name = format!("{:?}", call_inst.func);
        for (idx, id) in call_inst.ret_params.iter().enumerate() {
            let stays_as_return = match self.return_param_args.get(idx) {
                Some(x) => *x,
                None => return Err(TranslateError::MismatchedTypeIn(call_name.clone())),
            };
            if stays_as_return {
                if let Some(var) = func_decl_return_iter.next() {
                    return_arguments.push((*id, var.v_type.clone(), var.state_space));
                } else {
                    return Err(TranslateError::MismatchedTypeIn(call_name.clone()));
                }
            } else {
                if let Some(var) = func_decl_input_iter.next() {
                    input_arguments.push((
                        ast::Operand::Reg(*id),
                        var.v_type.clone(),
                        var.state_space,
                    ));
                } else {
                    return Err(TranslateError::MismatchedTypeIn(call_name.clone()));
                }
            }
        }
        if return_arguments.len() != func_decl.return_arguments.len()
            || input_arguments.len() != func_decl.input_arguments.len()
        {
            return Err(TranslateError::MismatchedTypeIn(call_name.clone()));
        }
        Ok(ResolvedCall {
            return_arguments,
            input_arguments,
            uniform: call_inst.uniform,
            name: call_inst.func,
        })
    }
}

struct GlobalStringIdResolver<'input> {
    current_id: spirv::Word,
    variables: HashMap<Cow<'input, str>, spirv::Word>,
    variables_type_check: HashMap<u32, Option<(ast::Type, ast::StateSpace, bool)>>,
    special_registers: SpecialRegistersMap,
    fns: HashMap<spirv::Word, FnSigMapper<'input>>,
}

impl<'input> GlobalStringIdResolver<'input> {
    fn new(start_id: spirv::Word) -> Self {
        Self {
            current_id: start_id,
            variables: HashMap::new(),
            variables_type_check: HashMap::new(),
            special_registers: SpecialRegistersMap::new(),
            fns: HashMap::new(),
        }
    }

    fn get_or_add_def(&mut self, id: &'input str) -> spirv::Word {
        self.get_or_add_impl(id, None)
    }

    fn get_or_add_def_typed(
        &mut self,
        id: &'input str,
        typ: ast::Type,
        state_space: ast::StateSpace,
        is_variable: bool,
    ) -> spirv::Word {
        self.get_or_add_impl(id, Some((typ, state_space, is_variable)))
    }

    fn get_or_add_impl(
        &mut self,
        id: &'input str,
        typ: Option<(ast::Type, ast::StateSpace, bool)>,
    ) -> spirv::Word {
        let id = match self.variables.entry(Cow::Borrowed(id)) {
            hash_map::Entry::Occupied(e) => *(e.get()),
            hash_map::Entry::Vacant(e) => {
                let numeric_id = self.current_id;
                e.insert(numeric_id);
                self.current_id += 1;
                numeric_id
            }
        };
        self.variables_type_check.insert(id, typ);
        id
    }

    fn get_id(&self, id: &str) -> Result<spirv::Word, TranslateError> {
        self.variables
            .get(id)
            .copied()
            .ok_or(TranslateError::UnknownSymbolName(id.to_string()))
    }

    fn current_id(&self) -> spirv::Word {
        self.current_id
    }

    fn start_fn<'b>(
        &'b mut self,
        header: &'b ast::MethodDeclaration<'input, &'input str>,
    ) -> Result<
        (
            FnStringIdResolver<'input, 'b>,
            GlobalFnDeclResolver<'input, 'b>,
            Rc<RefCell<ast::MethodDeclaration<'input, spirv::Word>>>,
        ),
        TranslateError,
    > {
        // In case a function decl was inserted earlier we want to use its id
        let name_id = self.get_or_add_def(header.name());
        let mut fn_resolver = FnStringIdResolver {
            current_id: &mut self.current_id,
            global_variables: &self.variables,
            global_type_check: &self.variables_type_check,
            special_registers: &mut self.special_registers,
            variables: vec![HashMap::new(); 1],
            type_check: HashMap::new(),
        };
        let return_arguments = rename_fn_params(&mut fn_resolver, &header.return_arguments);
        let input_arguments = rename_fn_params(&mut fn_resolver, &header.input_arguments);
        let name = match header.name {
            ast::MethodName::Kernel(name) => ast::MethodName::Kernel(name),
            ast::MethodName::Func(_) => ast::MethodName::Func(name_id),
        };
        let fn_decl = ast::MethodDeclaration {
            return_arguments,
            name,
            input_arguments,
            shared_mem: None,
            shared_mem_prefix: 0,
        };
        let new_fn_decl = if !fn_decl.name.is_kernel() {
            let resolver = FnSigMapper::remap_to_spirv_repr(fn_decl);
            let new_fn_decl = resolver.func_decl.clone();
            self.fns.insert(name_id, resolver);
            new_fn_decl
        } else {
            Rc::new(RefCell::new(fn_decl))
        };
        Ok((
            fn_resolver,
            GlobalFnDeclResolver { fns: &self.fns },
            new_fn_decl,
        ))
    }
}

pub struct GlobalFnDeclResolver<'input, 'a> {
    fns: &'a HashMap<spirv::Word, FnSigMapper<'input>>,
}

impl<'input, 'a> GlobalFnDeclResolver<'input, 'a> {
    fn get_fn_sig_resolver(&self, id: spirv::Word) -> Result<&FnSigMapper<'input>, TranslateError> {
        self.fns
            .get(&id)
            .ok_or(TranslateError::UnknownSymbolName(format!("fn#{}", id)))
    }
}

struct FnStringIdResolver<'input, 'b> {
    current_id: &'b mut spirv::Word,
    global_variables: &'b HashMap<Cow<'input, str>, spirv::Word>,
    global_type_check: &'b HashMap<u32, Option<(ast::Type, ast::StateSpace, bool)>>,
    special_registers: &'b mut SpecialRegistersMap,
    variables: Vec<HashMap<Cow<'input, str>, spirv::Word>>,
    type_check: HashMap<u32, Option<(ast::Type, ast::StateSpace, bool)>>,
}

impl<'a, 'b> FnStringIdResolver<'a, 'b> {
    fn finish(self) -> NumericIdResolver<'b> {
        NumericIdResolver {
            current_id: self.current_id,
            global_type_check: self.global_type_check,
            type_check: self.type_check,
            special_registers: self.special_registers,
        }
    }

    fn start_block(&mut self) {
        self.variables.push(HashMap::new())
    }

    fn end_block(&mut self) {
        self.variables.pop();
    }

    fn get_id(&mut self, id: &str) -> Result<spirv::Word, TranslateError> {
        for scope in self.variables.iter().rev() {
            match scope.get(id) {
                Some(id) => return Ok(*id),
                None => continue,
            }
        }
        match self.global_variables.get(id) {
            Some(found_id) => Ok(*found_id),
            None => {
                let sreg = PtxSpecialRegister::try_parse(id)
                    .ok_or_else(|| TranslateError::UnknownSymbolName(id.to_string()))?;
                Ok(self.special_registers.get_or_add(self.current_id, sreg))
            }
        }
    }

    fn add_def(
        &mut self,
        id: &'a str,
        typ: Option<(ast::Type, ast::StateSpace)>,
        is_variable: bool,
    ) -> spirv::Word {
        let numeric_id = *self.current_id;
        self.variables
            .last_mut()
            .unwrap()
            .insert(Cow::Borrowed(id), numeric_id);
        self.type_check.insert(
            numeric_id,
            typ.map(|(typ, space)| (typ, space, is_variable)),
        );
        *self.current_id += 1;
        numeric_id
    }

    #[must_use]
    fn add_defs(
        &mut self,
        base_id: &'a str,
        count: u32,
        typ: ast::Type,
        state_space: ast::StateSpace,
        is_variable: bool,
    ) -> impl Iterator<Item = spirv::Word> {
        let numeric_id = *self.current_id;
        for i in 0..count {
            self.variables
                .last_mut()
                .unwrap()
                .insert(Cow::Owned(format!("{}{}", base_id, i)), numeric_id + i);
            self.type_check.insert(
                numeric_id + i,
                Some((typ.clone(), state_space, is_variable)),
            );
        }
        *self.current_id += count;
        (0..count).into_iter().map(move |i| i + numeric_id)
    }
}

struct NumericIdResolver<'b> {
    current_id: &'b mut spirv::Word,
    global_type_check: &'b HashMap<u32, Option<(ast::Type, ast::StateSpace, bool)>>,
    type_check: HashMap<u32, Option<(ast::Type, ast::StateSpace, bool)>>,
    special_registers: &'b mut SpecialRegistersMap,
}

impl<'b> NumericIdResolver<'b> {
    fn finish(self) -> MutableNumericIdResolver<'b> {
        MutableNumericIdResolver { base: self }
    }

    fn get_typed(
        &self,
        id: spirv::Word,
    ) -> Result<(ast::Type, ast::StateSpace, bool), TranslateError> {
        match self.type_check.get(&id) {
            Some(Some(x)) => Ok(x.clone()),
            Some(None) => Err(TranslateError::UntypedSymbolName(id.to_string())),
            None => match self.special_registers.get(id) {
                Some(x) => Ok((x.get_type(), ast::StateSpace::Sreg, true)),
                None => match self.global_type_check.get(&id) {
                    Some(Some(result)) => Ok(result.clone()),
                    Some(None) | None => Err(TranslateError::UntypedSymbolName(id.to_string())),
                },
            },
        }
    }

    // This is for identifiers which will be emitted later as OpVariable
    // They are candidates for insertion of LoadVar/StoreVar
    fn register_variable(&mut self, typ: ast::Type, state_space: ast::StateSpace) -> spirv::Word {
        let new_id = *self.current_id;
        self.type_check
            .insert(new_id, Some((typ, state_space, true)));
        *self.current_id += 1;
        new_id
    }
    fn register_intermediate(&mut self, typ: Option<(ast::Type, ast::StateSpace)>) -> spirv::Word {
        let new_id = *self.current_id;
        self.type_check
            .insert(new_id, typ.map(|(t, space)| (t, space, false)));
        *self.current_id += 1;
        new_id
    }

}

struct MutableNumericIdResolver<'b> {
    base: NumericIdResolver<'b>,
}

impl<'b> MutableNumericIdResolver<'b> {
    fn unmut(self) -> NumericIdResolver<'b> {
        self.base
    }

    fn get_typed(&self, id: spirv::Word) -> Result<(ast::Type, ast::StateSpace), TranslateError> {
        self.base.get_typed(id).map(|(t, space, _)| (t, space))
    }

    fn register_intermediate(
        &mut self,
        typ: ast::Type,
        state_space: ast::StateSpace,
    ) -> spirv::Word {
        self.base.register_intermediate(Some((typ, state_space)))
    }
}

enum Statement<I, P: ast::ArgParams> {
    Label(u32),
    Variable(ast::Variable<P::Id>),
    Instruction(I),
    // SPIR-V compatible replacement for PTX predicates
    Conditional(BrachCondition),
    Call(ResolvedCall<P>),
    LoadVar(LoadVarDetails),
    StoreVar(StoreVarDetails),
    Conversion(ImplicitConversion),
    Constant(ConstantDefinition),
    RetValue(ast::RetData, spirv::Word),
    PtrAccess(PtrAccess<P>),
    RepackVector(RepackVectorDetails),
}

impl ExpandedStatement {
    fn map_id(self, f: &mut impl FnMut(spirv::Word, bool) -> spirv::Word) -> ExpandedStatement {
        match self {
            Statement::Label(id) => Statement::Label(f(id, false)),
            Statement::Variable(mut var) => {
                var.name = f(var.name, true);
                Statement::Variable(var)
            }
            Statement::Instruction(inst) => inst
                .visit(&mut |arg: ArgumentDescriptor<_>,
                             _: Option<(&ast::Type, ast::StateSpace)>| {
                    Ok(f(arg.op, arg.is_dst))
                })
                .unwrap(),
            Statement::LoadVar(mut details) => {
                details.arg.dst = f(details.arg.dst, true);
                details.arg.src = f(details.arg.src, false);
                Statement::LoadVar(details)
            }
            Statement::StoreVar(mut details) => {
                details.arg.src1 = f(details.arg.src1, false);
                details.arg.src2 = f(details.arg.src2, false);
                Statement::StoreVar(details)
            }
            Statement::Call(mut call) => {
                for (id, _, space) in call.return_arguments.iter_mut() {
                    let is_dst = match space {
                        ast::StateSpace::Reg => true,
                        ast::StateSpace::Param => false,
                        ast::StateSpace::Shared => false,
                        _ => todo!(),
                    };
                    *id = f(*id, is_dst);
                }
                call.name = f(call.name, false);
                for (id, _, _) in call.input_arguments.iter_mut() {
                    *id = f(*id, false);
                }
                Statement::Call(call)
            }
            Statement::Conditional(mut conditional) => {
                conditional.predicate = f(conditional.predicate, false);
                conditional.if_true = f(conditional.if_true, false);
                conditional.if_false = f(conditional.if_false, false);
                Statement::Conditional(conditional)
            }
            Statement::Conversion(mut conv) => {
                conv.dst = f(conv.dst, true);
                conv.src = f(conv.src, false);
                Statement::Conversion(conv)
            }
            Statement::Constant(mut constant) => {
                constant.dst = f(constant.dst, true);
                Statement::Constant(constant)
            }
            Statement::RetValue(data, id) => {
                let id = f(id, false);
                Statement::RetValue(data, id)
            }
            Statement::PtrAccess(PtrAccess {
                underlying_type,
                state_space,
                dst,
                ptr_src,
                offset_src: constant_src,
            }) => {
                let dst = f(dst, true);
                let ptr_src = f(ptr_src, false);
                let constant_src = f(constant_src, false);
                Statement::PtrAccess(PtrAccess {
                    underlying_type,
                    state_space,
                    dst,
                    ptr_src,
                    offset_src: constant_src,
                })
            }
            Statement::RepackVector(repack) => {
                let packed = f(repack.packed, !repack.is_extract);
                let unpacked = repack
                    .unpacked
                    .iter()
                    .map(|id| f(*id, repack.is_extract))
                    .collect();
                Statement::RepackVector(RepackVectorDetails {
                    packed,
                    unpacked,
                    ..repack
                })
            }
        }
    }
}

struct LoadVarDetails {
    arg: ast::Arg2<ExpandedArgParams>,
    typ: ast::Type,
    state_space: ast::StateSpace,
    // (index, vector_width)
    // IGC explodes when a builtin vector is loaded through
    // OpInBoundsAccessChain; it has to be OpLoad + OpCompositeExtract.
    member_index: Option<(u8, Option<u8>)>,
}

struct StoreVarDetails {
    arg: ast::Arg2St<ExpandedArgParams>,
    typ: ast::Type,
    member_index: Option<u8>,
}

struct RepackVectorDetails {
    is_extract: bool,
    typ: ast::ScalarType,
    packed: spirv::Word,
    unpacked: Vec<spirv::Word>,
    /// Number of elements in the pack. A pack of two 32-bit registers can also
    /// spell a single 64-bit value (`mov.b64 {%r0,%r1}, %rd2;`), which is then
    /// assembled into a 64-bit value instead of a vector.
    pack_width: u8,
    /// The type of the packed value when it is a scalar the pack splits into
    /// pieces. None when the packed value is a vector.
    packed_scalar: Option<ast::ScalarType>,
    non_default_implicit_conversion: Option<
        fn(
            (ast::StateSpace, &ast::Type),
            (ast::StateSpace, &ast::Type),
        ) -> Result<Option<ConversionKind>, TranslateError>,
    >,
}

impl RepackVectorDetails {
    /// The type of the packed value: a vector for a real vector pack, a scalar
    /// for the "the pack names the pieces of a wider value" form.
    fn packed_type(&self) -> ast::Type {
        match self.packed_scalar {
            Some(scalar) => ast::Type::Scalar(scalar),
            None => ast::Type::Vector(self.typ, self.unpacked.len() as u8),
        }
    }

    fn map<
        From: ArgParamsEx<Id = spirv::Word>,
        To: ArgParamsEx<Id = spirv::Word>,
        V: ArgumentMapVisitor<From, To>,
    >(
        self,
        visitor: &mut V,
    ) -> Result<RepackVectorDetails, TranslateError> {
        let scalar = visitor.id(
            ArgumentDescriptor {
                op: self.packed,
                is_dst: !self.is_extract,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((&self.packed_type(), ast::StateSpace::Reg)),
        )?;
        let scalar_type = self.typ;
        let is_extract = self.is_extract;
        let non_default_implicit_conversion = self.non_default_implicit_conversion;
        let vector = self
            .unpacked
            .into_iter()
            .map(|id| {
                if id == 0 {
                    // A hole in the pack: the element is not written.
                    return Ok(0);
                }
                visitor.id(
                    ArgumentDescriptor {
                        op: id,
                        is_dst: is_extract,
                        is_memory_access: false,
                        non_default_implicit_conversion,
                    },
                    Some((&ast::Type::Scalar(scalar_type), ast::StateSpace::Reg)),
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(RepackVectorDetails {
            is_extract,
            typ: self.typ,
            packed: scalar,
            unpacked: vector,
            pack_width: self.pack_width,
            packed_scalar: self.packed_scalar,
            non_default_implicit_conversion,
        })
    }
}

impl<T: ArgParamsEx<Id = spirv::Word>, U: ArgParamsEx<Id = spirv::Word>> Visitable<T, U>
    for RepackVectorDetails
{
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<T, U>,
    ) -> Result<Statement<ast::Instruction<U>, U>, TranslateError> {
        Ok(Statement::RepackVector(self.map::<_, _, _>(visitor)?))
    }
}

struct ResolvedCall<P: ast::ArgParams> {
    pub uniform: bool,
    pub return_arguments: Vec<(P::Id, ast::Type, ast::StateSpace)>,
    pub name: P::Id,
    pub input_arguments: Vec<(P::Operand, ast::Type, ast::StateSpace)>,
}

impl<T: ast::ArgParams> ResolvedCall<T> {
    fn cast<U: ast::ArgParams<Id = T::Id, Operand = T::Operand>>(self) -> ResolvedCall<U> {
        ResolvedCall {
            uniform: self.uniform,
            return_arguments: self.return_arguments,
            name: self.name,
            input_arguments: self.input_arguments,
        }
    }
}

impl<From: ArgParamsEx<Id = spirv::Word>> ResolvedCall<From> {
    fn map<To: ArgParamsEx<Id = spirv::Word>, V: ArgumentMapVisitor<From, To>>(
        self,
        visitor: &mut V,
    ) -> Result<ResolvedCall<To>, TranslateError> {
        let return_arguments = self
            .return_arguments
            .into_iter()
            .map::<Result<_, TranslateError>, _>(|(id, typ, space)| {
                let new_id = visitor.id(
                    ArgumentDescriptor {
                        op: id,
                        is_dst: space != ast::StateSpace::Param,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    Some((&typ, space)),
                )?;
                Ok((new_id, typ, space))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let func = visitor.id(
            ArgumentDescriptor {
                op: self.name,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            None,
        )?;
        let input_arguments = self
            .input_arguments
            .into_iter()
            .map::<Result<_, TranslateError>, _>(|(id, typ, space)| {
                let new_id = visitor.operand(
                    ArgumentDescriptor {
                        op: id,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &typ,
                    space,
                )?;
                Ok((new_id, typ, space))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ResolvedCall {
            uniform: self.uniform,
            return_arguments,
            name: func,
            input_arguments,
        })
    }
}

impl<T: ArgParamsEx<Id = spirv::Word>, U: ArgParamsEx<Id = spirv::Word>> Visitable<T, U>
    for ResolvedCall<T>
{
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<T, U>,
    ) -> Result<Statement<ast::Instruction<U>, U>, TranslateError> {
        Ok(Statement::Call(self.map(visitor)?))
    }
}

impl<P: ArgParamsEx<Id = spirv::Word>> PtrAccess<P> {
    fn map<To: ArgParamsEx<Id = spirv::Word>, V: ArgumentMapVisitor<P, To>>(
        self,
        visitor: &mut V,
    ) -> Result<PtrAccess<To>, TranslateError> {
        let new_dst = visitor.id(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((&self.underlying_type, self.state_space)),
        )?;
        let new_ptr_src = visitor.id(
            ArgumentDescriptor {
                op: self.ptr_src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((&self.underlying_type, self.state_space)),
        )?;
        let new_constant_src = visitor.operand(
            ArgumentDescriptor {
                op: self.offset_src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::S64),
            ast::StateSpace::Reg,
        )?;
        Ok(PtrAccess {
            underlying_type: self.underlying_type,
            state_space: self.state_space,
            dst: new_dst,
            ptr_src: new_ptr_src,
            offset_src: new_constant_src,
        })
    }
}

impl<T: ArgParamsEx<Id = spirv::Word>, U: ArgParamsEx<Id = spirv::Word>> Visitable<T, U>
    for PtrAccess<T>
{
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<T, U>,
    ) -> Result<Statement<ast::Instruction<U>, U>, TranslateError> {
        Ok(Statement::PtrAccess(self.map(visitor)?))
    }
}

pub trait ArgParamsEx: ast::ArgParams + Sized {}

impl<'input> ArgParamsEx for ast::ParsedArgParams<'input> {}

enum NormalizedArgParams {}

impl ast::ArgParams for NormalizedArgParams {
    type Id = spirv::Word;
    type Operand = ast::Operand<spirv::Word>;
}

impl ArgParamsEx for NormalizedArgParams {}

type NormalizedStatement = Statement<
    (
        Option<ast::PredAt<spirv::Word>>,
        ast::Instruction<NormalizedArgParams>,
    ),
    NormalizedArgParams,
>;

type UnconditionalStatement = Statement<ast::Instruction<NormalizedArgParams>, NormalizedArgParams>;

enum TypedArgParams {}

impl ast::ArgParams for TypedArgParams {
    type Id = spirv::Word;
    type Operand = TypedOperand;
}

impl ArgParamsEx for TypedArgParams {}

#[derive(Copy, Clone)]
enum TypedOperand {
    Reg(spirv::Word),
    RegOffset(spirv::Word, i32),
    Imm(ast::ImmediateValue),
    VecMember(spirv::Word, u8),
}

impl TypedOperand {
    fn upcast(self) -> ast::Operand<spirv::Word> {
        match self {
            TypedOperand::Reg(reg) => ast::Operand::Reg(reg),
            TypedOperand::RegOffset(reg, idx) => ast::Operand::RegOffset(reg, idx),
            TypedOperand::Imm(x) => ast::Operand::Imm(x),
            TypedOperand::VecMember(vec, idx) => ast::Operand::VecMember(vec, idx),
        }
    }
}

type TypedStatement = Statement<ast::Instruction<TypedArgParams>, TypedArgParams>;

enum ExpandedArgParams {}
type ExpandedStatement = Statement<ast::Instruction<ExpandedArgParams>, ExpandedArgParams>;

impl ast::ArgParams for ExpandedArgParams {
    type Id = spirv::Word;
    type Operand = spirv::Word;
}

impl ArgParamsEx for ExpandedArgParams {}

enum Directive<'input> {
    Variable(ast::LinkingDirective, ast::Variable<spirv::Word>),
    Method(Function<'input>),
}

struct Function<'input> {
    pub func_decl: Rc<RefCell<ast::MethodDeclaration<'input, spirv::Word>>>,
    pub globals: Vec<ast::Variable<spirv::Word>>,
    pub body: Option<Vec<ExpandedStatement>>,
    import_as: Option<String>,
    tuning: Vec<ast::TuningDirective>,
}

pub trait ArgumentMapVisitor<T: ArgParamsEx, U: ArgParamsEx> {
    fn id(
        &mut self,
        desc: ArgumentDescriptor<T::Id>,
        typ: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<U::Id, TranslateError>;
    fn operand(
        &mut self,
        desc: ArgumentDescriptor<T::Operand>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<U::Operand, TranslateError>;
}

impl<T> ArgumentMapVisitor<ExpandedArgParams, ExpandedArgParams> for T
where
    T: FnMut(
        ArgumentDescriptor<spirv::Word>,
        Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError>,
{
    fn id(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        t: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        self(desc, t)
    }

    fn operand(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<spirv::Word, TranslateError> {
        self(desc, Some((typ, state_space)))
    }
}

impl<'a, T> ArgumentMapVisitor<ast::ParsedArgParams<'a>, NormalizedArgParams> for T
where
    T: FnMut(&str) -> Result<spirv::Word, TranslateError>,
{
    fn id(
        &mut self,
        desc: ArgumentDescriptor<&str>,
        _: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        self(desc.op)
    }

    fn operand(
        &mut self,
        desc: ArgumentDescriptor<ast::Operand<&str>>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<ast::Operand<spirv::Word>, TranslateError> {
        Ok(match desc.op {
            ast::Operand::Reg(id) => ast::Operand::Reg(self(id)?),
            ast::Operand::RegOffset(id, imm) => ast::Operand::RegOffset(self(id)?, imm),
            ast::Operand::Imm(imm) => ast::Operand::Imm(imm),
            ast::Operand::VecMember(id, member) => ast::Operand::VecMember(self(id)?, member),
            ast::Operand::VecPack(ref ids) => ast::Operand::VecPack(
                ids.into_iter()
                    .map(|id| {
                        if *id == "_" {
                            // A hole in the pack: the element is not written.
                            // 0 is never a valid SPIR-V id.
                            Ok(0)
                        } else {
                            self.id(desc.new_op(id), Some((typ, state_space)))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ast::Operand::MixedPack(elements) => ast::Operand::MixedPack(
                elements
                    .into_iter()
                    .map(|element| match element {
                        ast::PackElement::Reg(id) => Ok(ast::PackElement::Reg(self(id)?)),
                        ast::PackElement::Imm(value) => Ok(ast::PackElement::Imm(value)),
                    })
                    .collect::<Result<_, TranslateError>>()?,
            ),
            // A vector literal of constants: "st.global.v2.b32 [addr], {99, 0};"
            // Nothing to resolve here.
            ast::Operand::ImmPack(values) => ast::Operand::ImmPack(values),
        })
    }
}

pub struct ArgumentDescriptor<Op> {
    op: Op,
    is_dst: bool,
    is_memory_access: bool,
    non_default_implicit_conversion: Option<
        fn(
            (ast::StateSpace, &ast::Type),
            (ast::StateSpace, &ast::Type),
        ) -> Result<Option<ConversionKind>, TranslateError>,
    >,
}

pub struct PtrAccess<P: ast::ArgParams> {
    underlying_type: ast::Type,
    state_space: ast::StateSpace,
    dst: spirv::Word,
    ptr_src: spirv::Word,
    offset_src: P::Operand,
}

impl<T> ArgumentDescriptor<T> {
    fn new_op<U>(&self, u: U) -> ArgumentDescriptor<U> {
        ArgumentDescriptor {
            op: u,
            is_dst: self.is_dst,
            is_memory_access: self.is_memory_access,
            non_default_implicit_conversion: self.non_default_implicit_conversion,
        }
    }
}

impl<T: ArgParamsEx> ast::Instruction<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
    ) -> Result<ast::Instruction<U>, TranslateError> {
        Ok(match self {
            ast::Instruction::Abs(d, arg) => {
                ast::Instruction::Abs(d, arg.map(visitor, &ast::Type::Scalar(d.typ))?)
            }
            // Call instruction is converted to a call statement early on
            ast::Instruction::Call(_) => return Err(error_unreachable()),
            ast::Instruction::Ld(d, a) => {
                let new_args = a.map(visitor, &d)?;
                ast::Instruction::Ld(d, new_args)
            }
            ast::Instruction::Mov(d, a) => {
                let mapped = a.map(visitor, &d)?;
                ast::Instruction::Mov(d, mapped)
            }
            ast::Instruction::CpAsync(d, a) => {
                // Both operands are plain addresses in registers: the shared
                // one is an offset into the shared window, the global one a
                // byte address.
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &ast::Type::Scalar(ast::ScalarType::B32),
                    ast::StateSpace::Reg,
                )?;
                let src = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &ast::Type::Scalar(ast::ScalarType::B64),
                    ast::StateSpace::Reg,
                )?;
                let src_size = match a.src_size {
                    Some(size) => Some(visitor.operand(
                        ArgumentDescriptor {
                            op: size,
                            is_dst: false,
                            is_memory_access: false,
                            non_default_implicit_conversion: None,
                        },
                        &ast::Type::Scalar(ast::ScalarType::B32),
                        ast::StateSpace::Reg,
                    )?),
                    None => None,
                };
                ast::Instruction::CpAsync(
                    d,
                    ast::ArgCpAsync {
                        dst,
                        src,
                        src_size,
                    },
                )
            }
            ast::Instruction::Trap => ast::Instruction::Trap,
            ast::Instruction::BrxIdx(index, table, targets) => {
                let u32 = ast::Type::Scalar(ast::ScalarType::U32);
                let index = visitor.operand(
                    ArgumentDescriptor {
                        op: index,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &u32,
                    ast::StateSpace::Reg,
                )?;
                // The table is a label, not a value: it carries no type and
                // takes part in no conversion, so the operand field is filled
                // with the index (the emitter reads only the targets).
                ast::Instruction::BrxIdx(index, None, targets)
            }
            ast::Instruction::VSub4(a) => {
                let b32 = ast::Type::Scalar(ast::ScalarType::B32);
                let map_arg = |op, is_dst, visitor: &mut V| -> Result<U::Operand, TranslateError> {
                    visitor.operand(
                        ArgumentDescriptor {
                            op,
                            is_dst,
                            is_memory_access: false,
                            non_default_implicit_conversion: None,
                        },
                        &b32,
                        ast::StateSpace::Reg,
                    )
                };
                ast::Instruction::VSub4(ast::Arg3 {
                    dst: map_arg(a.dst, true, visitor)?,
                    src1: map_arg(a.src1, false, visitor)?,
                    src2: map_arg(a.src2, false, visitor)?,
                })
            }
            ast::Instruction::VSet4(op, a) => {
                let b32 = ast::Type::Scalar(ast::ScalarType::B32);
                let map_arg = |op, is_dst, visitor: &mut V| -> Result<U::Operand, TranslateError> {
                    visitor.operand(
                        ArgumentDescriptor {
                            op,
                            is_dst,
                            is_memory_access: false,
                            non_default_implicit_conversion: None,
                        },
                        &b32,
                        ast::StateSpace::Reg,
                    )
                };
                ast::Instruction::VSet4(
                    op,
                    ast::Arg3 {
                        dst: map_arg(a.dst, true, visitor)?,
                        src1: map_arg(a.src1, false, visitor)?,
                        src2: map_arg(a.src2, false, visitor)?,
                    },
                )
            }
            ast::Instruction::Redux(op, typ, a) => {
                let value_type = ast::Type::Scalar(typ);
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &value_type,
                    ast::StateSpace::Reg,
                )?;
                let src1 = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src1,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &value_type,
                    ast::StateSpace::Reg,
                )?;
                let src2 = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src2,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &ast::Type::Scalar(ast::ScalarType::B32),
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::Redux(op, typ, ast::Arg3 { dst, src1, src2 })
            }
            ast::Instruction::CvtPacked(typ, a) => {
                // One 32-bit register holds the two converted elements.
                let b32 = ast::Type::Scalar(ast::ScalarType::B32);
                let f32 = ast::Type::Scalar(ast::ScalarType::F32);
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &b32,
                    ast::StateSpace::Reg,
                )?;
                let src1 = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src1,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &f32,
                    ast::StateSpace::Reg,
                )?;
                let src2 = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src2,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &f32,
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::CvtPacked(typ, ast::Arg3 { dst, src1, src2 })
            }
            ast::Instruction::MovMatrix(a) => {
                // Both operands are one 32-bit register holding two halves.
                let b32 = ast::Type::Scalar(ast::ScalarType::B32);
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &b32,
                    ast::StateSpace::Reg,
                )?;
                let src = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &b32,
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::MovMatrix(ast::Arg2 { dst, src })
            }
            ast::Instruction::Mma(d, a) => {
                // Every thread owns four elements of D (and of C): four floats,
                // four integers, or two packed pairs of halves - a half
                // precision accumulation keeps two elements per register.
                let result_bits: u32 = match d.dst {
                    ast::ScalarType::F32 | ast::ScalarType::S32 | ast::ScalarType::U32 => 32,
                    ast::ScalarType::F16 | ast::ScalarType::S16 | ast::ScalarType::U16 => 16,
                    _ => return Err(error_todo()),
                };
                let result_registers =
                    ((d.m as u32 * d.n as u32 / 32) * result_bits / 32) as u8;
                let result_type = if d.dst == ast::ScalarType::F32 {
                    ast::ScalarType::F32
                } else {
                    ast::ScalarType::B32
                };
                // The fragment sizes follow from the shape and the element
                // width: a thread holds m*k/32 elements of A, k*n/32 of B and
                // m*n/32 of D, and a register holds 32 bits of them.
                let element_bits: u8 = match d.a {
                    ast::ScalarType::F16 | ast::ScalarType::B16 => 16,
                    ast::ScalarType::B32 => 32,
                    ast::ScalarType::S8 | ast::ScalarType::U8 => 8,
                    _ => return Err(error_todo()),
                };
                let a_registers = ((d.m as u32 * d.k as u32 / 32) * element_bits as u32 / 32) as u8;
                let b_registers = ((d.n as u32 * d.k as u32 / 32) * element_bits as u32 / 32) as u8;
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &if result_registers == 1 {
                        ast::Type::Scalar(result_type)
                    } else {
                        ast::Type::Vector(result_type, result_registers)
                    },
                    ast::StateSpace::Reg,
                )?;
                let a_regs = visitor.operand(
                    ArgumentDescriptor {
                        op: a.a,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &if a_registers == 1 {
                        ast::Type::Scalar(ast::ScalarType::B32)
                    } else {
                        ast::Type::Vector(ast::ScalarType::B32, a_registers)
                    },
                    ast::StateSpace::Reg,
                )?;
                let b_regs = visitor.operand(
                    ArgumentDescriptor {
                        op: a.b,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &if b_registers == 1 {
                        ast::Type::Scalar(ast::ScalarType::B32)
                    } else {
                        ast::Type::Vector(ast::ScalarType::B32, b_registers)
                    },
                    ast::StateSpace::Reg,
                )?;
                let c_regs = visitor.operand(
                    ArgumentDescriptor {
                        op: a.c,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &if result_registers == 1 {
                        ast::Type::Scalar(result_type)
                    } else {
                        ast::Type::Vector(result_type, result_registers)
                    },
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::Mma(
                    d,
                    ast::ArgMma {
                        dst,
                        a: a_regs,
                        b: b_regs,
                        c: c_regs,
                    },
                )
            }
            ast::Instruction::Shf(d, a) => {
                ast::Instruction::Shf(d, a.map(visitor, &ast::Type::Scalar(ast::ScalarType::B32), false)?)
            }
            ast::Instruction::Copysign(t, a) => {
                ast::Instruction::Copysign(
                    t,
                    a.map_non_shift(visitor, &ast::Type::Scalar(t), false)?,
                )
            }
            ast::Instruction::Dp4a(t, a) => {
                ast::Instruction::Dp4a(t, a.map(visitor, &ast::Type::Scalar(t), false)?)
            }
            ast::Instruction::Prmt(t, a) => {
                ast::Instruction::Prmt(t, a.map(visitor, &ast::Type::Scalar(t), false)?)
            }
            ast::Instruction::Nanosleep(a) => ast::Instruction::Nanosleep(
                a.map(visitor, Some((&ast::Type::Scalar(ast::ScalarType::U32), ast::StateSpace::Reg)))?,
            ),
            ast::Instruction::CpAsyncCommitGroup => ast::Instruction::CpAsyncCommitGroup,
            ast::Instruction::CpAsyncWaitGroup(n) => ast::Instruction::CpAsyncWaitGroup(n),
            ast::Instruction::CpAsyncWaitAll => ast::Instruction::CpAsyncWaitAll,
            ast::Instruction::LdMatrix(d, a) => {
                // The fragments are a vector of 32-bit registers, the address
                // a 64-bit value.
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &if d.num == 1 {
                        ast::Type::Scalar(ast::ScalarType::B32)
                    } else {
                        ast::Type::Vector(ast::ScalarType::B32, d.num)
                    },
                    ast::StateSpace::Reg,
                )?;
                let src = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src,
                        is_dst: false,
                        is_memory_access: true,
                        non_default_implicit_conversion: None,
                    },
                    &ast::Type::Scalar(ast::ScalarType::B64),
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::LdMatrix(d, ast::Arg2LdMatrix { dst, src })
            }
            ast::Instruction::Mul(d, a) => {
                let inst_type = d.get_type();
                let is_wide = d.is_wide();
                ast::Instruction::Mul(d, a.map_non_shift(visitor, &inst_type, is_wide)?)
            }
            ast::Instruction::Add(d, a) => {
                let inst_type = d.get_type();
                ast::Instruction::Add(d, a.map_non_shift(visitor, &inst_type, false)?)
            }
            ast::Instruction::Setp(d, a) => {
                let inst_type = d.typ;
                ast::Instruction::Setp(d, a.map(visitor, &ast::Type::Scalar(inst_type))?)
            }
            ast::Instruction::SetpBool(d, a) => {
                let inst_type = d.typ;
                ast::Instruction::SetpBool(d, a.map(visitor, &ast::Type::Scalar(inst_type))?)
            }
            ast::Instruction::Not(t, a) => {
                ast::Instruction::Not(t, a.map(visitor, &ast::Type::Scalar(t))?)
            }
            ast::Instruction::Cvt(d, a) => {
                let (dst_t, src_t) = match &d {
                    ast::CvtDetails::FloatFromFloat(desc) => (
                        ast::Type::Scalar(desc.dst.into()),
                        ast::Type::Scalar(desc.src.into()),
                    ),
                    ast::CvtDetails::FloatFromInt(desc) => (
                        ast::Type::Scalar(desc.dst.into()),
                        ast::Type::Scalar(desc.src.into()),
                    ),
                    ast::CvtDetails::IntFromFloat(desc) => (
                        ast::Type::Scalar(desc.dst.into()),
                        ast::Type::Scalar(desc.src.into()),
                    ),
                    ast::CvtDetails::IntFromInt(desc) => (
                        ast::Type::Scalar(desc.dst.into()),
                        ast::Type::Scalar(desc.src.into()),
                    ),
                };
                ast::Instruction::Cvt(d, a.map_different_types(visitor, &dst_t, &src_t)?)
            }
            ast::Instruction::Shl(t, a) => {
                ast::Instruction::Shl(t, a.map_shift(visitor, &ast::Type::Scalar(t))?)
            }
            ast::Instruction::Shr(t, a) => {
                ast::Instruction::Shr(t, a.map_shift(visitor, &ast::Type::Scalar(t.into()))?)
            }
            ast::Instruction::St(d, a) => {
                let new_args = a.map(visitor, &d)?;
                ast::Instruction::St(d, new_args)
            }
            ast::Instruction::Bra(d, a) => ast::Instruction::Bra(d, a.map(visitor, None)?),
            ast::Instruction::Ret(d) => ast::Instruction::Ret(d),
            ast::Instruction::Cvta(d, a) => {
                let inst_type = ast::Type::Scalar(ast::ScalarType::B64);
                ast::Instruction::Cvta(d, a.map(visitor, &inst_type)?)
            }
            ast::Instruction::Mad(d, a) => {
                let inst_type = d.get_type();
                let is_wide = d.is_wide();
                ast::Instruction::Mad(d, a.map(visitor, &inst_type, is_wide)?)
            }
            ast::Instruction::Or(t, a) => ast::Instruction::Or(
                t,
                a.map_non_shift(visitor, &ast::Type::Scalar(t.into()), false)?,
            ),
            ast::Instruction::Sub(d, a) => {
                let typ = d.get_type();
                ast::Instruction::Sub(d, a.map_non_shift(visitor, &typ, false)?)
            }
            ast::Instruction::Min(d, a) => {
                let typ = d.get_type();
                ast::Instruction::Min(d, a.map_non_shift(visitor, &typ, false)?)
            }
            ast::Instruction::Max(d, a) => {
                let typ = d.get_type();
                ast::Instruction::Max(d, a.map_non_shift(visitor, &typ, false)?)
            }
            ast::Instruction::Rcp(d, a) => {
                let typ = ast::Type::Scalar(if d.is_f64 {
                    ast::ScalarType::F64
                } else {
                    ast::ScalarType::F32
                });
                ast::Instruction::Rcp(d, a.map(visitor, &typ)?)
            }
            ast::Instruction::And(t, a) => ast::Instruction::And(
                t,
                a.map_non_shift(visitor, &ast::Type::Scalar(t.into()), false)?,
            ),
            ast::Instruction::Selp(t, a) => ast::Instruction::Selp(t, a.map_selp(visitor, t)?),
            ast::Instruction::Bar(d, a) => ast::Instruction::Bar(d, a.map(visitor)?),
            ast::Instruction::Membar(scope) => ast::Instruction::Membar(scope),
            ast::Instruction::Shfl(m, t, a) => {
                let typ = ast::Type::Scalar(t);
                // The optional destination predicate is a predicate register.
                let dst_predicate = match a.dst_predicate {
                    Some(op) => Some(visitor.operand(
                        ArgumentDescriptor {
                            op,
                            is_dst: true,
                            is_memory_access: false,
                            non_default_implicit_conversion: None,
                        },
                        &ast::Type::Scalar(ast::ScalarType::Pred),
                        ast::StateSpace::Reg,
                    )?),
                    None => None,
                };
                let mut map_operand = |op: T::Operand, is_dst: bool| {
                    visitor.operand(
                        ArgumentDescriptor {
                            op,
                            is_dst,
                            is_memory_access: false,
                            non_default_implicit_conversion: None,
                        },
                        &typ,
                        ast::StateSpace::Reg,
                    )
                };
                ast::Instruction::Shfl(
                    m,
                    t,
                    ast::ArgShfl {
                        dst: map_operand(a.dst, true)?,
                        dst_predicate,
                        src: map_operand(a.src, false)?,
                        lane: map_operand(a.lane, false)?,
                        clamp: map_operand(a.clamp, false)?,
                        member_mask: map_operand(a.member_mask, false)?,
                    },
                )
            }
            ast::Instruction::Activemask(dst) => {
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::Activemask(dst)
            }
            ast::Instruction::Vote(m, t, a) => {
                let typ = ast::Type::Scalar(t);
                let pred_typ = ast::Type::Scalar(ast::ScalarType::Pred);
                // The destination keeps the instruction type (b32 for the ballot
                // form, pred for the reduction forms), the source is always a
                // predicate and the member mask a 32-bit integer.
                let dst = visitor.operand(
                    ArgumentDescriptor {
                        op: a.dst,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &typ,
                    ast::StateSpace::Reg,
                )?;
                let src = visitor.operand(
                    ArgumentDescriptor {
                        op: a.src,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &pred_typ,
                    ast::StateSpace::Reg,
                )?;
                let member_mask = visitor.operand(
                    ArgumentDescriptor {
                        op: a.member_mask,
                        is_dst: false,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    &ast::Type::Scalar(ast::ScalarType::U32),
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::Vote(
                    m,
                    t,
                    ast::ArgVote {
                        dst,
                        src,
                        member_mask,
                    },
                )
            }
            ast::Instruction::BarRedPred(and, d, b, p) => {
                let pred_t = &ast::Type::Scalar(ast::ScalarType::Pred);
                let b32_t = &ast::Type::Scalar(ast::ScalarType::B32);
                let new_d = visitor.operand(
                    ArgumentDescriptor { op: d, is_dst: true, is_memory_access: false, non_default_implicit_conversion: None },
                    pred_t,
                    ast::StateSpace::Reg,
                )?;
                let new_b = visitor.operand(
                    ArgumentDescriptor { op: b, is_dst: false, is_memory_access: false, non_default_implicit_conversion: None },
                    b32_t,
                    ast::StateSpace::Reg,
                )?;
                let new_p = visitor.operand(
                    ArgumentDescriptor { op: p, is_dst: false, is_memory_access: false, non_default_implicit_conversion: None },
                    pred_t,
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::BarRedPred(and, new_d, new_b, new_p)
            }
            ast::Instruction::BarRedPopc(d, b, p) => {
                let t = &ast::Type::Scalar(ast::ScalarType::U32);
                let pred_t = &ast::Type::Scalar(ast::ScalarType::Pred);
                let new_d = visitor.operand(
                    ArgumentDescriptor { op: d, is_dst: true, is_memory_access: false, non_default_implicit_conversion: None },
                    t,
                    ast::StateSpace::Reg,
                )?;
                let new_b = visitor.operand(
                    ArgumentDescriptor { op: b, is_dst: false, is_memory_access: false, non_default_implicit_conversion: None },
                    t,
                    ast::StateSpace::Reg,
                )?;
                let new_p = visitor.operand(
                    ArgumentDescriptor { op: p, is_dst: false, is_memory_access: false, non_default_implicit_conversion: None },
                    pred_t,
                    ast::StateSpace::Reg,
                )?;
                ast::Instruction::BarRedPopc(new_d, new_b, new_p)
            }
            ast::Instruction::Atom(d, a) => {
                ast::Instruction::Atom(d, a.map_atom(visitor, d.inner.get_type(), d.space)?)
            }
            ast::Instruction::Red(d, a) => {
                ast::Instruction::Red(d, a.map_atom(visitor, d.inner.get_type(), d.space)?)
            }
            ast::Instruction::AtomCas(d, a) => {
                ast::Instruction::AtomCas(d, a.map_atom(visitor, d.typ, d.space)?)
            }
            ast::Instruction::Div(d, a) => {
                ast::Instruction::Div(d, a.map_non_shift(visitor, &d.get_type(), false)?)
            }
            ast::Instruction::Sqrt(d, a) => {
                ast::Instruction::Sqrt(d, a.map(visitor, &ast::Type::Scalar(d.typ.into()))?)
            }
            ast::Instruction::Rsqrt(d, a) => {
                ast::Instruction::Rsqrt(d, a.map(visitor, &ast::Type::Scalar(d.typ.into()))?)
            }
            ast::Instruction::Neg(d, a) => {
                ast::Instruction::Neg(d, a.map(visitor, &ast::Type::Scalar(d.typ))?)
            }
            ast::Instruction::Sin { flush_to_zero, arg } => {
                let typ = ast::Type::Scalar(ast::ScalarType::F32);
                ast::Instruction::Sin {
                    flush_to_zero,
                    arg: arg.map(visitor, &typ)?,
                }
            }
            ast::Instruction::Cos { flush_to_zero, arg } => {
                let typ = ast::Type::Scalar(ast::ScalarType::F32);
                ast::Instruction::Cos {
                    flush_to_zero,
                    arg: arg.map(visitor, &typ)?,
                }
            }
            ast::Instruction::Lg2 { flush_to_zero, arg } => {
                let typ = ast::Type::Scalar(ast::ScalarType::F32);
                ast::Instruction::Lg2 {
                    flush_to_zero,
                    arg: arg.map(visitor, &typ)?,
                }
            }
            ast::Instruction::Ex2 { flush_to_zero, arg } => {
                let typ = ast::Type::Scalar(ast::ScalarType::F32);
                ast::Instruction::Ex2 {
                    flush_to_zero,
                    arg: arg.map(visitor, &typ)?,
                }
            }
            ast::Instruction::Clz { typ, arg } => {
                let dst_type = ast::Type::Scalar(ast::ScalarType::B32);
                let src_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Clz {
                    typ,
                    arg: arg.map_different_types(visitor, &dst_type, &src_type)?,
                }
            }
            ast::Instruction::Bfind { typ, arg } => {
                let dst_type = ast::Type::Scalar(ast::ScalarType::B32);
                let src_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Bfind {
                    typ,
                    arg: arg.map_different_types(visitor, &dst_type, &src_type)?,
                }
            }
            ast::Instruction::Brev { typ, arg } => {
                let full_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Brev {
                    typ,
                    arg: arg.map(visitor, &full_type)?,
                }
            }
            ast::Instruction::Popc { typ, arg } => {
                let dst_type = ast::Type::Scalar(ast::ScalarType::B32);
                let src_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Popc {
                    typ,
                    arg: arg.map_different_types(visitor, &dst_type, &src_type)?,
                }
            }
            ast::Instruction::Xor { typ, arg } => {
                let full_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Xor {
                    typ,
                    arg: arg.map_non_shift(visitor, &full_type, false)?,
                }
            }
            ast::Instruction::Bfe { typ, arg } => {
                let full_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Bfe {
                    typ,
                    arg: arg.map_bfe(visitor, &full_type)?,
                }
            }
            ast::Instruction::Bfi { typ, arg } => {
                let full_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Bfi {
                    typ,
                    arg: arg.map_bfi(visitor, &full_type)?,
                }
            }
            ast::Instruction::Rem { typ, arg } => {
                let full_type = ast::Type::Scalar(typ.into());
                ast::Instruction::Rem {
                    typ,
                    arg: arg.map_non_shift(visitor, &full_type, false)?,
                }
            }
        })
    }
}

impl<T: ArgParamsEx, U: ArgParamsEx> Visitable<T, U> for ast::Instruction<T> {
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<T, U>,
    ) -> Result<Statement<ast::Instruction<U>, U>, TranslateError> {
        Ok(Statement::Instruction(self.map(visitor)?))
    }
}

impl ImplicitConversion {
    fn map<
        T: ArgParamsEx<Id = spirv::Word>,
        U: ArgParamsEx<Id = spirv::Word>,
        V: ArgumentMapVisitor<T, U>,
    >(
        self,
        visitor: &mut V,
    ) -> Result<Statement<ast::Instruction<U>, U>, TranslateError> {
        let new_dst = visitor.id(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((&self.to_type, self.to_space)),
        )?;
        let new_src = visitor.id(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((&self.from_type, self.from_space)),
        )?;
        Ok(Statement::Conversion({
            ImplicitConversion {
                src: new_src,
                dst: new_dst,
                ..self
            }
        }))
    }
}

impl<From: ArgParamsEx<Id = spirv::Word>, To: ArgParamsEx<Id = spirv::Word>> Visitable<From, To>
    for ImplicitConversion
{
    fn visit(
        self,
        visitor: &mut impl ArgumentMapVisitor<From, To>,
    ) -> Result<Statement<ast::Instruction<To>, To>, TranslateError> {
        Ok(self.map(visitor)?)
    }
}

impl<T> ArgumentMapVisitor<TypedArgParams, TypedArgParams> for T
where
    T: FnMut(
        ArgumentDescriptor<spirv::Word>,
        Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError>,
{
    fn id(
        &mut self,
        desc: ArgumentDescriptor<spirv::Word>,
        t: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<spirv::Word, TranslateError> {
        self(desc, t)
    }

    fn operand(
        &mut self,
        desc: ArgumentDescriptor<TypedOperand>,
        typ: &ast::Type,
        state_space: ast::StateSpace,
    ) -> Result<TypedOperand, TranslateError> {
        Ok(match desc.op {
            TypedOperand::Reg(id) => {
                TypedOperand::Reg(self(desc.new_op(id), Some((typ, state_space)))?)
            }
            TypedOperand::Imm(imm) => TypedOperand::Imm(imm),
            TypedOperand::RegOffset(id, imm) => {
                TypedOperand::RegOffset(self(desc.new_op(id), Some((typ, state_space)))?, imm)
            }
            TypedOperand::VecMember(reg, index) => {
                let scalar_type = match typ {
                    ast::Type::Scalar(scalar_t) => *scalar_t,
                    _ => return Err(error_unreachable()),
                };
                let vec_type = ast::Type::Vector(scalar_type, index + 1);
                TypedOperand::VecMember(
                    self(desc.new_op(reg), Some((&vec_type, state_space)))?,
                    index,
                )
            }
        })
    }
}

impl ast::Type {
    fn widen(self) -> Result<Self, TranslateError> {
        match self {
            ast::Type::Scalar(scalar) => {
                let kind = scalar.kind();
                let width = scalar.size_of();
                if (kind != ast::ScalarKind::Signed
                    && kind != ast::ScalarKind::Unsigned
                    && kind != ast::ScalarKind::Bit)
                    || (width == 8)
                {
                    return Err(error_mismatched_type());
                }
                let width = scalar.size_of();
                if (kind != ast::ScalarKind::Signed
                    && kind != ast::ScalarKind::Unsigned
                    && kind != ast::ScalarKind::Bit)
                    || (width == 8)
                {
                    return Err(error_mismatched_type());
                }
                Ok(ast::Type::Scalar(ast::ScalarType::from_parts(
                    width * 2,
                    kind,
                )))
            }
            _ => Err(error_unreachable()),
        }
    }

    fn to_parts(&self) -> TypeParts {
        match self {
            ast::Type::Scalar(scalar) => TypeParts {
                kind: TypeKind::Scalar,
                state_space: ast::StateSpace::Reg,
                scalar_kind: scalar.kind(),
                width: scalar.size_of(),
                components: Vec::new(),
            },
            ast::Type::Vector(scalar, components) => TypeParts {
                kind: TypeKind::Vector,
                state_space: ast::StateSpace::Reg,
                scalar_kind: scalar.kind(),
                width: scalar.size_of(),
                components: vec![*components as u32],
            },
            ast::Type::Array(scalar, components) => TypeParts {
                kind: TypeKind::Array,
                state_space: ast::StateSpace::Reg,
                scalar_kind: scalar.kind(),
                width: scalar.size_of(),
                components: components.clone(),
            },
            ast::Type::Struct(members) => TypeParts {
                kind: TypeKind::Array,
                state_space: ast::StateSpace::Reg,
                scalar_kind: ast::ScalarKind::Unsigned,
                width: 4,
                components: vec![members.len() as u32],
            },
            ast::Type::Pointer(scalar, space) => TypeParts {
                kind: TypeKind::Pointer,
                state_space: *space,
                scalar_kind: scalar.kind(),
                width: scalar.size_of(),
                components: Vec::new(),
            },
        }
    }

    fn from_parts(t: TypeParts) -> Self {
        match t.kind {
            TypeKind::Scalar => {
                ast::Type::Scalar(ast::ScalarType::from_parts(t.width, t.scalar_kind))
            }
            TypeKind::Vector => ast::Type::Vector(
                ast::ScalarType::from_parts(t.width, t.scalar_kind),
                t.components[0] as u8,
            ),
            TypeKind::Array => ast::Type::Array(
                ast::ScalarType::from_parts(t.width, t.scalar_kind),
                t.components,
            ),
            TypeKind::Pointer => ast::Type::Pointer(
                ast::ScalarType::from_parts(t.width, t.scalar_kind),
                t.state_space,
            ),
        }
    }

    pub fn size_of(&self) -> usize {
        match self {
            ast::Type::Scalar(typ) => typ.size_of() as usize,
            ast::Type::Vector(typ, len) => (typ.size_of() as usize) * (*len as usize),
            ast::Type::Array(typ, len) => len
                .iter()
                .fold(typ.size_of() as usize, |x, y| (x as usize) * (*y as usize)),
            ast::Type::Struct(members) => 4 * members.len(),
            ast::Type::Pointer(..) => mem::size_of::<usize>(),
        }
    }
}

#[derive(Eq, PartialEq, Clone)]
struct TypeParts {
    kind: TypeKind,
    scalar_kind: ast::ScalarKind,
    width: u8,
    state_space: ast::StateSpace,
    components: Vec<u32>,
}

#[derive(Eq, PartialEq, Copy, Clone)]
enum TypeKind {
    Scalar,
    Vector,
    Array,
    Pointer,
}

impl ast::Instruction<ExpandedArgParams> {
    fn jump_target(&self) -> Option<spirv::Word> {
        match self {
            ast::Instruction::Bra(_, a) => Some(a.src),
            _ => None,
        }
    }

    // .wide instructions don't support ftz, so it's enough to just look at the
    // type declared by the instruction
    fn flush_to_zero(&self) -> Option<(bool, u8)> {
        match self {
            ast::Instruction::Ld(_, _) => None,
            ast::Instruction::St(_, _) => None,
            ast::Instruction::LdMatrix(_, _) => None,
            ast::Instruction::CpAsync(_, _) => None,
            ast::Instruction::Prmt(_, _) => None,
            ast::Instruction::Dp4a(_, _) => None,
            // copysign is a bit operation, the rounding of the input is the
            // caller's business.
            ast::Instruction::Copysign(_, _) => None,
            ast::Instruction::Trap => None,
            ast::Instruction::Shf(_, _) => None,
            ast::Instruction::Mma(_, _) => None,
            ast::Instruction::MovMatrix(_) => None,
            ast::Instruction::CvtPacked(_, _) => None,
            ast::Instruction::Redux(_, _, _) => None,
            ast::Instruction::VSub4(_) => None,
            ast::Instruction::VSet4(_, _) => None,
            ast::Instruction::BrxIdx(_, _, _) => None,
            ast::Instruction::Nanosleep(_) => None,
            ast::Instruction::CpAsyncCommitGroup => None,
            ast::Instruction::CpAsyncWaitGroup(_) => None,
            ast::Instruction::CpAsyncWaitAll => None,
            ast::Instruction::Mov(_, _) => None,
            ast::Instruction::Not(_, _) => None,
            ast::Instruction::Bra(_, _) => None,
            ast::Instruction::Shl(_, _) => None,
            ast::Instruction::Shr(_, _) => None,
            ast::Instruction::Ret(_) => None,
            ast::Instruction::Call(_) => None,
            ast::Instruction::Or(_, _) => None,
            ast::Instruction::And(_, _) => None,
            ast::Instruction::Cvta(_, _) => None,
            ast::Instruction::Selp(_, _) => None,
            ast::Instruction::Bar(_, _) => None,
            ast::Instruction::Membar(_) => None,
            ast::Instruction::BarRedPopc(_, _, _) => None,
            ast::Instruction::BarRedPred(_, _, _, _) => None,
            ast::Instruction::Shfl(_, _, _) => None,
            ast::Instruction::Vote(_, _, _) => None,
            ast::Instruction::Activemask(_) => None,
            ast::Instruction::Atom(_, _) => None,
            ast::Instruction::Red(_, _) => None,
            ast::Instruction::AtomCas(_, _) => None,
            ast::Instruction::Sub(ast::ArithDetails::Signed(_), _) => None,
            ast::Instruction::Sub(ast::ArithDetails::Unsigned(_), _) => None,
            ast::Instruction::Add(ast::ArithDetails::Signed(_), _) => None,
            ast::Instruction::Add(ast::ArithDetails::Unsigned(_), _) => None,
            ast::Instruction::Mul(ast::MulDetails::Unsigned(_), _) => None,
            ast::Instruction::Mul(ast::MulDetails::Signed(_), _) => None,
            ast::Instruction::Mad(ast::MulDetails::Unsigned(_), _) => None,
            ast::Instruction::Mad(ast::MulDetails::Signed(_), _) => None,
            ast::Instruction::Min(ast::MinMaxDetails::Signed(_), _) => None,
            ast::Instruction::Min(ast::MinMaxDetails::Unsigned(_), _) => None,
            ast::Instruction::Max(ast::MinMaxDetails::Signed(_), _) => None,
            ast::Instruction::Max(ast::MinMaxDetails::Unsigned(_), _) => None,
            ast::Instruction::Cvt(ast::CvtDetails::IntFromInt(_), _) => None,
            ast::Instruction::Cvt(ast::CvtDetails::FloatFromInt(_), _) => None,
            ast::Instruction::Div(ast::DivDetails::Unsigned(_), _) => None,
            ast::Instruction::Div(ast::DivDetails::Signed(_), _) => None,
            ast::Instruction::Clz { .. } => None,
            ast::Instruction::Bfind { .. } => None,
            ast::Instruction::Brev { .. } => None,
            ast::Instruction::Popc { .. } => None,
            ast::Instruction::Xor { .. } => None,
            ast::Instruction::Bfe { .. } => None,
            ast::Instruction::Bfi { .. } => None,
            ast::Instruction::Rem { .. } => None,
            ast::Instruction::Sub(ast::ArithDetails::Float(float_control), _)
            | ast::Instruction::Add(ast::ArithDetails::Float(float_control), _)
            | ast::Instruction::Mul(ast::MulDetails::Float(float_control), _)
            | ast::Instruction::Mad(ast::MulDetails::Float(float_control), _) => float_control
                .flush_to_zero
                .map(|ftz| (ftz, ast::ScalarType::from(float_control.typ).size_of())),
            ast::Instruction::Setp(details, _) => details
                .flush_to_zero
                .map(|ftz| (ftz, details.typ.size_of())),
            ast::Instruction::SetpBool(details, _) => details
                .flush_to_zero
                .map(|ftz| (ftz, details.typ.size_of())),
            ast::Instruction::Abs(details, _) => details
                .flush_to_zero
                .map(|ftz| (ftz, details.typ.size_of())),
            ast::Instruction::Min(ast::MinMaxDetails::Float(float_control), _)
            | ast::Instruction::Max(ast::MinMaxDetails::Float(float_control), _) => float_control
                .flush_to_zero
                .map(|ftz| (ftz, ast::ScalarType::from(float_control.typ).size_of())),
            ast::Instruction::Rcp(details, _) => details
                .flush_to_zero
                .map(|ftz| (ftz, if details.is_f64 { 8 } else { 4 })),
            // Modifier .ftz can only be specified when either .dtype or .atype
            // is .f32 and applies only to single precision (.f32) inputs and results.
            ast::Instruction::Cvt(
                ast::CvtDetails::FloatFromFloat(ast::CvtDesc { flush_to_zero, .. }),
                _,
            )
            | ast::Instruction::Cvt(
                ast::CvtDetails::IntFromFloat(ast::CvtDesc { flush_to_zero, .. }),
                _,
            ) => flush_to_zero.map(|ftz| (ftz, 4)),
            ast::Instruction::Div(ast::DivDetails::Float(details), _) => details
                .flush_to_zero
                .map(|ftz| (ftz, ast::ScalarType::from(details.typ).size_of())),
            ast::Instruction::Sqrt(details, _) => details
                .flush_to_zero
                .map(|ftz| (ftz, ast::ScalarType::from(details.typ).size_of())),
            ast::Instruction::Rsqrt(details, _) => Some((
                details.flush_to_zero,
                ast::ScalarType::from(details.typ).size_of(),
            )),
            ast::Instruction::Neg(details, _) => details
                .flush_to_zero
                .map(|ftz| (ftz, details.typ.size_of())),
            ast::Instruction::Sin { flush_to_zero, .. }
            | ast::Instruction::Cos { flush_to_zero, .. }
            | ast::Instruction::Lg2 { flush_to_zero, .. }
            | ast::Instruction::Ex2 { flush_to_zero, .. } => {
                Some((*flush_to_zero, mem::size_of::<f32>() as u8))
            }
        }
    }
}

type Arg2 = ast::Arg2<ExpandedArgParams>;
type Arg2St = ast::Arg2St<ExpandedArgParams>;

struct ConstantDefinition {
    pub dst: spirv::Word,
    pub typ: ast::ScalarType,
    pub value: ast::ImmediateValue,
}

struct BrachCondition {
    predicate: spirv::Word,
    if_true: spirv::Word,
    if_false: spirv::Word,
}

#[derive(Clone)]
struct ImplicitConversion {
    src: spirv::Word,
    dst: spirv::Word,
    from_type: ast::Type,
    to_type: ast::Type,
    from_space: ast::StateSpace,
    to_space: ast::StateSpace,
    kind: ConversionKind,
}

#[derive(PartialEq, Clone)]
enum ConversionKind {
    Default,
    SignExtend,
    // The value is wider than the instruction wants: keep the low bits.
    Truncate,
    BitToPtr,
    PtrToPtr,
    AddressOf,
}

impl<T> ast::PredAt<T> {
    fn map_variable<U, F: FnMut(T) -> Result<U, TranslateError>>(
        self,
        f: &mut F,
    ) -> Result<ast::PredAt<U>, TranslateError> {
        let new_label = f(self.label)?;
        Ok(ast::PredAt {
            not: self.not,
            label: new_label,
        })
    }
}

impl<'a> ast::Instruction<ast::ParsedArgParams<'a>> {
    fn map_variable<F: FnMut(&str) -> Result<spirv::Word, TranslateError>>(
        self,
        f: &mut F,
    ) -> Result<ast::Instruction<NormalizedArgParams>, TranslateError> {
        match self {
            ast::Instruction::Call(call) => {
                let call_inst = ast::CallInst {
                    uniform: call.uniform,
                    ret_params: call
                        .ret_params
                        .into_iter()
                        .map(|p| f(p))
                        .collect::<Result<_, _>>()?,
                    func: f(call.func)?,
                    param_list: call
                        .param_list
                        .into_iter()
                        .map(|p| p.map_variable(f))
                        .collect::<Result<_, _>>()?,
                };
                Ok(ast::Instruction::Call(call_inst))
            }
            i => i.map(f),
        }
    }
}

impl<T: ArgParamsEx> ast::Arg1<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: Option<(&ast::Type, ast::StateSpace)>,
    ) -> Result<ast::Arg1<U>, TranslateError> {
        let new_src = visitor.id(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
        )?;
        Ok(ast::Arg1 { src: new_src })
    }
}

impl<T: ArgParamsEx> ast::Arg1Bar<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
    ) -> Result<ast::Arg1Bar<U>, TranslateError> {
        let new_src = visitor.operand(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::U32),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg1Bar { src: new_src })
    }
}

impl<T: ArgParamsEx> ast::Arg2<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: &ast::Type,
    ) -> Result<ast::Arg2<U>, TranslateError> {
        let new_dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let new_src = visitor.operand(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg2 {
            dst: new_dst,
            src: new_src,
        })
    }

    fn map_different_types<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        dst_t: &ast::Type,
        src_t: &ast::Type,
    ) -> Result<ast::Arg2<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            dst_t,
            ast::StateSpace::Reg,
        )?;
        let src = visitor.operand(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: Some(should_convert_relaxed_src_wrapper),
            },
            src_t,
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg2 { dst, src })
    }
}

impl<T: ArgParamsEx> ast::Arg2Ld<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        details: &ast::LdDetails,
    ) -> Result<ast::Arg2Ld<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: Some(should_convert_relaxed_dst_wrapper),
            },
            &ast::Type::from(details.typ.clone()),
            ast::StateSpace::Reg,
        )?;
        let src = visitor.operand(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: true,
                non_default_implicit_conversion: None,
            },
            &details.typ,
            details.state_space,
        )?;
        Ok(ast::Arg2Ld { dst, src })
    }
}

impl<T: ArgParamsEx> ast::Arg2St<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        details: &ast::StData,
    ) -> Result<ast::Arg2St<U>, TranslateError> {
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: true,
                non_default_implicit_conversion: None,
            },
            &details.typ,
            details.state_space,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: Some(should_convert_relaxed_src_wrapper),
            },
            &details.typ.clone().into(),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg2St { src1, src2 })
    }
}

impl<T: ArgParamsEx> ast::Arg2St<T> {
    fn map_atom<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: ast::ScalarType,
        state_space: ast::StateSpace,
    ) -> Result<ast::Arg2St<U>, TranslateError> {
        let scalar_type = ast::ScalarType::from(t);
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: true,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            state_space,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg2St { src1, src2 })
    }
}

impl<T: ArgParamsEx> ast::Arg2Mov<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        details: &ast::MovDetails,
    ) -> Result<ast::Arg2Mov<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &details.typ.clone().into(),
            ast::StateSpace::Reg,
        )?;
        let src = visitor.operand(
            ArgumentDescriptor {
                op: self.src,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: Some(implicit_conversion_mov),
            },
            &details.typ.clone().into(),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg2Mov { dst, src })
    }
}

impl<T: ArgParamsEx> ast::Arg3<T> {
    fn map_non_shift<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        typ: &ast::Type,
        is_wide: bool,
    ) -> Result<ast::Arg3<U>, TranslateError> {
        let wide_type = if is_wide {
            Some(typ.clone().widen()?)
        } else {
            None
        };
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            wide_type.as_ref().unwrap_or(typ),
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            typ,
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            typ,
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg3 { dst, src1, src2 })
    }

    fn map_shift<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: &ast::Type,
    ) -> Result<ast::Arg3<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::U32),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg3 { dst, src1, src2 })
    }

    fn map_atom<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: ast::ScalarType,
        state_space: ast::StateSpace,
    ) -> Result<ast::Arg3<U>, TranslateError> {
        let scalar_type = ast::ScalarType::from(t);
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: true,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            state_space,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg3 { dst, src1, src2 })
    }
}

impl<T: ArgParamsEx> ast::Arg4<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: &ast::Type,
        is_wide: bool,
    ) -> Result<ast::Arg4<U>, TranslateError> {
        let wide_type = if is_wide {
            Some(t.clone().widen()?)
        } else {
            None
        };
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            wide_type.as_ref().unwrap_or(t),
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src3 = visitor.operand(
            ArgumentDescriptor {
                op: self.src3,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg4 {
            dst,
            src1,
            src2,
            src3,
        })
    }

    fn map_selp<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: ast::ScalarType,
    ) -> Result<ast::Arg4<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(t.into()),
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(t.into()),
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(t.into()),
            ast::StateSpace::Reg,
        )?;
        let src3 = visitor.operand(
            ArgumentDescriptor {
                op: self.src3,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::Pred),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg4 {
            dst,
            src1,
            src2,
            src3,
        })
    }

    fn map_atom<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: ast::ScalarType,
        state_space: ast::StateSpace,
    ) -> Result<ast::Arg4<U>, TranslateError> {
        let scalar_type = ast::ScalarType::from(t);
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: true,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            state_space,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            ast::StateSpace::Reg,
        )?;
        let src3 = visitor.operand(
            ArgumentDescriptor {
                op: self.src3,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(scalar_type),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg4 {
            dst,
            src1,
            src2,
            src3,
        })
    }

    fn map_bfe<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        typ: &ast::Type,
    ) -> Result<ast::Arg4<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            typ,
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            typ,
            ast::StateSpace::Reg,
        )?;
        // The second operand is compared against the first, so it has the
        // comparison's type: typing it as u32 made a `.s16` comparison against
        // -1 emit an out-of-range 16-bit constant.
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            typ,
            ast::StateSpace::Reg,
        )?;
        // The offset and the width are operands of the extraction, which wants
        // them at the width of the result: a 64-bit bfe with a 32-bit width is
        // rejected by the backend.
        let src3 = visitor.operand(
            ArgumentDescriptor {
                op: self.src3,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            typ,
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg4 {
            dst,
            src1,
            src2,
            src3,
        })
    }
}

impl<T: ArgParamsEx> ast::Arg4Setp<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: &ast::Type,
    ) -> Result<ast::Arg4Setp<U>, TranslateError> {
        let dst1 = visitor.id(
            ArgumentDescriptor {
                op: self.dst1,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((
                &ast::Type::Scalar(ast::ScalarType::Pred),
                ast::StateSpace::Reg,
            )),
        )?;
        let dst2 = self
            .dst2
            .map(|dst2| {
                visitor.id(
                    ArgumentDescriptor {
                        op: dst2,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::Pred),
                        ast::StateSpace::Reg,
                    )),
                )
            })
            .transpose()?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg4Setp {
            dst1,
            dst2,
            src1,
            src2,
        })
    }
}

impl<T: ArgParamsEx> ast::Arg5<T> {
    fn map_bfi<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        base_type: &ast::Type,
    ) -> Result<ast::Arg5<U>, TranslateError> {
        let dst = visitor.operand(
            ArgumentDescriptor {
                op: self.dst,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            base_type,
            ast::StateSpace::Reg,
        )?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            base_type,
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            base_type,
            ast::StateSpace::Reg,
        )?;
        let src3 = visitor.operand(
            ArgumentDescriptor {
                op: self.src3,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::U32),
            ast::StateSpace::Reg,
        )?;
        let src4 = visitor.operand(
            ArgumentDescriptor {
                op: self.src4,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::U32),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg5 {
            dst,
            src1,
            src2,
            src3,
            src4,
        })
    }
}

impl<T: ArgParamsEx> ast::Arg5Setp<T> {
    fn map<U: ArgParamsEx, V: ArgumentMapVisitor<T, U>>(
        self,
        visitor: &mut V,
        t: &ast::Type,
    ) -> Result<ast::Arg5Setp<U>, TranslateError> {
        let dst1 = visitor.id(
            ArgumentDescriptor {
                op: self.dst1,
                is_dst: true,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            Some((
                &ast::Type::Scalar(ast::ScalarType::Pred),
                ast::StateSpace::Reg,
            )),
        )?;
        let dst2 = self
            .dst2
            .map(|dst2| {
                visitor.id(
                    ArgumentDescriptor {
                        op: dst2,
                        is_dst: true,
                        is_memory_access: false,
                        non_default_implicit_conversion: None,
                    },
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::Pred),
                        ast::StateSpace::Reg,
                    )),
                )
            })
            .transpose()?;
        let src1 = visitor.operand(
            ArgumentDescriptor {
                op: self.src1,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src2 = visitor.operand(
            ArgumentDescriptor {
                op: self.src2,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            t,
            ast::StateSpace::Reg,
        )?;
        let src3 = visitor.operand(
            ArgumentDescriptor {
                op: self.src3,
                is_dst: false,
                is_memory_access: false,
                non_default_implicit_conversion: None,
            },
            &ast::Type::Scalar(ast::ScalarType::Pred),
            ast::StateSpace::Reg,
        )?;
        Ok(ast::Arg5Setp {
            dst1,
            dst2,
            src1,
            src2,
            src3,
        })
    }
}

impl<T> ast::Operand<T> {
    fn map_variable<U, F: FnMut(T) -> Result<U, TranslateError>>(
        self,
        f: &mut F,
    ) -> Result<ast::Operand<U>, TranslateError> {
        Ok(match self {
            ast::Operand::Reg(reg) => ast::Operand::Reg(f(reg)?),
            ast::Operand::RegOffset(reg, offset) => ast::Operand::RegOffset(f(reg)?, offset),
            ast::Operand::Imm(x) => ast::Operand::Imm(x),
            ast::Operand::VecMember(reg, idx) => ast::Operand::VecMember(f(reg)?, idx),
            ast::Operand::VecPack(vec) => {
                ast::Operand::VecPack(vec.into_iter().map(f).collect::<Result<_, _>>()?)
            }
            ast::Operand::ImmPack(values) => ast::Operand::ImmPack(values),
            ast::Operand::MixedPack(elements) => ast::Operand::MixedPack(
                elements
                    .into_iter()
                    .map(|element| match element {
                        ast::PackElement::Reg(id) => Ok(ast::PackElement::Reg(f(id)?)),
                        ast::PackElement::Imm(value) => Ok(ast::PackElement::Imm(value)),
                    })
                    .collect::<Result<_, TranslateError>>()?,
            ),
        })
    }
}

impl ast::Operand<spirv::Word> {
    fn unwrap_reg(&self) -> Result<spirv::Word, TranslateError> {
        match self {
            ast::Operand::Reg(reg) => Ok(*reg),
            _ => Err(error_unreachable()),
        }
    }
}

impl ast::ScalarType {
    fn from_parts(width: u8, kind: ast::ScalarKind) -> Self {
        match kind {
            ast::ScalarKind::Float => match width {
                2 => ast::ScalarType::F16,
                4 => ast::ScalarType::F32,
                8 => ast::ScalarType::F64,
                _ => unreachable!(),
            },
            ast::ScalarKind::Bit => match width {
                1 => ast::ScalarType::B8,
                2 => ast::ScalarType::B16,
                4 => ast::ScalarType::B32,
                8 => ast::ScalarType::B64,
                _ => unreachable!(),
            },
            ast::ScalarKind::Signed => match width {
                1 => ast::ScalarType::S8,
                2 => ast::ScalarType::S16,
                4 => ast::ScalarType::S32,
                8 => ast::ScalarType::S64,
                _ => unreachable!(),
            },
            ast::ScalarKind::Unsigned => match width {
                1 => ast::ScalarType::U8,
                2 => ast::ScalarType::U16,
                4 => ast::ScalarType::U32,
                8 => ast::ScalarType::U64,
                _ => unreachable!(),
            },
            ast::ScalarKind::Float2 => match width {
                4 => ast::ScalarType::F16x2,
                _ => unreachable!(),
            },
            ast::ScalarKind::Pred => ast::ScalarType::Pred,
        }
    }
}

impl ast::ArithDetails {
    fn get_type(&self) -> ast::Type {
        ast::Type::Scalar(match self {
            ast::ArithDetails::Unsigned(t) => (*t).into(),
            ast::ArithDetails::Signed(d) => d.typ.into(),
            ast::ArithDetails::Float(d) => d.typ.into(),
        })
    }
}

impl ast::MulDetails {
    fn get_type(&self) -> ast::Type {
        ast::Type::Scalar(match self {
            ast::MulDetails::Unsigned(d) => d.typ.into(),
            ast::MulDetails::Signed(d) => d.typ.into(),
            ast::MulDetails::Float(d) => d.typ.into(),
        })
    }
}

impl ast::MinMaxDetails {
    fn get_type(&self) -> ast::Type {
        ast::Type::Scalar(match self {
            ast::MinMaxDetails::Signed(t) => (*t).into(),
            ast::MinMaxDetails::Unsigned(t) => (*t).into(),
            ast::MinMaxDetails::Float(d) => d.typ.into(),
        })
    }
}

impl ast::DivDetails {
    fn get_type(&self) -> ast::Type {
        ast::Type::Scalar(match self {
            ast::DivDetails::Unsigned(t) => (*t).into(),
            ast::DivDetails::Signed(t) => (*t).into(),
            ast::DivDetails::Float(d) => d.typ.into(),
        })
    }
}

impl ast::AtomInnerDetails {
    fn get_type(&self) -> ast::ScalarType {
        match self {
            ast::AtomInnerDetails::Bit { typ, .. } => (*typ).into(),
            ast::AtomInnerDetails::Unsigned { typ, .. } => (*typ).into(),
            ast::AtomInnerDetails::Signed { typ, .. } => (*typ).into(),
            ast::AtomInnerDetails::Float { typ, .. } => (*typ).into(),
        }
    }
}

impl ast::StateSpace {
    fn to_spirv(self) -> spirv::StorageClass {
        match self {
            ast::StateSpace::Const => spirv::StorageClass::UniformConstant,
            ast::StateSpace::Generic => spirv::StorageClass::Generic,
            ast::StateSpace::Global => spirv::StorageClass::CrossWorkgroup,
            ast::StateSpace::Local => spirv::StorageClass::Function,
            ast::StateSpace::Shared => spirv::StorageClass::Workgroup,
            ast::StateSpace::Param => spirv::StorageClass::Function,
            ast::StateSpace::Reg => spirv::StorageClass::Function,
            ast::StateSpace::Sreg => spirv::StorageClass::Input,
        }
    }

    fn is_compatible(self, other: ast::StateSpace) -> bool {
        self == other
            || self == ast::StateSpace::Reg && other == ast::StateSpace::Sreg
            || self == ast::StateSpace::Sreg && other == ast::StateSpace::Reg
    }

    fn coerces_to_generic(self) -> bool {
        match self {
            ast::StateSpace::Global
            | ast::StateSpace::Const
            | ast::StateSpace::Local
            | ast::StateSpace::Shared => true,
            ast::StateSpace::Reg
            | ast::StateSpace::Param
            | ast::StateSpace::Generic
            | ast::StateSpace::Sreg => false,
        }
    }

    fn is_addressable(self) -> bool {
        match self {
            ast::StateSpace::Const
            | ast::StateSpace::Generic
            | ast::StateSpace::Global
            | ast::StateSpace::Local
            | ast::StateSpace::Shared => true,
            ast::StateSpace::Param | ast::StateSpace::Reg | ast::StateSpace::Sreg => false,
        }
    }
}

impl<T> ast::Operand<T> {
    fn underlying_register(&self) -> Option<&T> {
        match self {
            ast::Operand::Reg(r)
            | ast::Operand::RegOffset(r, _)
            | ast::Operand::VecMember(r, _) => Some(r),
            ast::Operand::Imm(_)
            | ast::Operand::VecPack(..)
            | ast::Operand::ImmPack(..)
            | ast::Operand::MixedPack(..) => None,
        }
    }
}

impl ast::MulDetails {
    fn is_wide(&self) -> bool {
        match self {
            ast::MulDetails::Unsigned(d) => d.control == ast::MulIntControl::Wide,
            ast::MulDetails::Signed(d) => d.control == ast::MulIntControl::Wide,
            ast::MulDetails::Float(_) => false,
        }
    }
}

impl ast::MemScope {
    fn to_spirv(self) -> spirv::Scope {
        match self {
            ast::MemScope::Cta => spirv::Scope::Workgroup,
            ast::MemScope::Gpu => spirv::Scope::Device,
            ast::MemScope::Sys => spirv::Scope::CrossDevice,
        }
    }
}

impl ast::AtomSemantics {
    fn to_spirv(self) -> spirv::MemorySemantics {
        match self {
            ast::AtomSemantics::Relaxed => spirv::MemorySemantics::RELAXED,
            ast::AtomSemantics::Acquire => spirv::MemorySemantics::ACQUIRE,
            ast::AtomSemantics::Release => spirv::MemorySemantics::RELEASE,
            ast::AtomSemantics::AcquireRelease => spirv::MemorySemantics::ACQUIRE_RELEASE,
        }
    }
}

fn default_implicit_conversion(
    (operand_space, operand_type): (ast::StateSpace, &ast::Type),
    (instruction_space, instruction_type): (ast::StateSpace, &ast::Type),
) -> Result<Option<ConversionKind>, TranslateError> {
    if !instruction_space.is_compatible(operand_space) {
        default_implicit_conversion_space(
            (operand_space, operand_type),
            (instruction_space, instruction_type),
        )
    } else if instruction_type != operand_type {
        default_implicit_conversion_type(instruction_space, operand_type, instruction_type)
    } else {
        Ok(None)
    }
}

// Space is different
fn default_implicit_conversion_space(
    (operand_space, operand_type): (ast::StateSpace, &ast::Type),
    (instruction_space, instruction_type): (ast::StateSpace, &ast::Type),
) -> Result<Option<ConversionKind>, TranslateError> {
    if (instruction_space == ast::StateSpace::Generic && operand_space.coerces_to_generic())
        || (operand_space == ast::StateSpace::Generic && instruction_space.coerces_to_generic())
    {
        Ok(Some(ConversionKind::PtrToPtr))
    } else if operand_space.is_compatible(ast::StateSpace::Reg) {
        match operand_type {
            ast::Type::Pointer(operand_ptr_type, operand_ptr_space)
                if *operand_ptr_space == instruction_space =>
            {
                if instruction_type != &ast::Type::Scalar(*operand_ptr_type) {
                    Ok(Some(ConversionKind::PtrToPtr))
                } else {
                    Ok(None)
                }
            }
            // TODO: 32 bit
            ast::Type::Scalar(ast::ScalarType::B64)
            | ast::Type::Scalar(ast::ScalarType::U64)
            | ast::Type::Scalar(ast::ScalarType::S64) => match instruction_space {
                ast::StateSpace::Global
                | ast::StateSpace::Generic
                | ast::StateSpace::Const
                | ast::StateSpace::Local
                | ast::StateSpace::Shared
                | ast::StateSpace::Param
                // A 64-bit register holding an address is the normal way PTX
                // spells a generic pointer.
                | ast::StateSpace::Reg => Ok(Some(ConversionKind::BitToPtr)),
                _ => {
                    Err(error_mismatched_type())
                }
            },
            ast::Type::Scalar(ast::ScalarType::B32)
            | ast::Type::Scalar(ast::ScalarType::U32)
            | ast::Type::Scalar(ast::ScalarType::S32) => match instruction_space {
                ast::StateSpace::Const
                | ast::StateSpace::Local
                | ast::StateSpace::Shared
                // A 32-bit integer used as an address is legal in PTX (it is
                // zero-extended); nvcc emits it for global/shared pointers
                // that are known to fit in 32 bits.
                | ast::StateSpace::Global
                | ast::StateSpace::Generic
                | ast::StateSpace::Param => Ok(Some(ConversionKind::BitToPtr)),
                _ => Err(error_mismatched_type()),
            },
            _ => {
                Err(error_mismatched_type())
            }
        }
    } else if instruction_space.is_compatible(ast::StateSpace::Reg) {
        match instruction_type {
            ast::Type::Pointer(instruction_ptr_type, instruction_ptr_space)
                if operand_space == *instruction_ptr_space =>
            {
                if operand_type != &ast::Type::Scalar(*instruction_ptr_type) {
                    Ok(Some(ConversionKind::PtrToPtr))
                } else {
                    Ok(None)
                }
            }
            _ => Err(error_mismatched_type()),
        }
    } else {
        Err(error_mismatched_type())
    }
}

// Space is same, but type is different
fn default_implicit_conversion_type(
    space: ast::StateSpace,
    operand_type: &ast::Type,
    instruction_type: &ast::Type,
) -> Result<Option<ConversionKind>, TranslateError> {
    if space.is_compatible(ast::StateSpace::Reg) {
        // The instruction wants a narrower value than the register holds, or has
        // written one (an 8-bit value lives in the low byte of a 16-bit
        // register): keep the low bits.
        //
        // The register's own declaration does not decide this. PTX lets a narrow
        // result be written to a register declared `.s16` or `.u16` as well as
        // `.b16`, and the hardware extends it by the sign of the INSTRUCTION's
        // type: `cvt.s8.s32` leaves 0xff80 in all three, and `cvt.u8.u32` leaves
        // 0x0080 in all three. Only a register declared `.bN` used to be
        // accepted here, so the same instruction into a `.s16` register failed
        // to translate at all.
        if let (ast::Type::Scalar(instr), ast::Type::Scalar(operand)) =
            (instruction_type, operand_type)
        {
            if instr.size_of() < operand.size_of()
                && operand.kind().is_integer()
                && instr.kind().is_integer()
            {
                return Ok(Some(ConversionKind::Truncate));
            }
        }
        if should_bitcast(instruction_type, operand_type) {
            Ok(Some(ConversionKind::Default))
        } else {
            Err(TranslateError::MismatchedTypeIn(format!(
                "instruction wants {}, operand is {}",
                describe_type(instruction_type),
                describe_type(operand_type)
            )))
        }
    } else {
        Ok(Some(ConversionKind::PtrToPtr))
    }
}

fn should_bitcast(instr: &ast::Type, operand: &ast::Type) -> bool {
    match (instr, operand) {
        (ast::Type::Scalar(inst), ast::Type::Scalar(operand)) => {
            if inst.size_of() != operand.size_of() {
                return false;
            }
            match inst.kind() {
                ast::ScalarKind::Bit => operand.kind() != ast::ScalarKind::Bit,
                ast::ScalarKind::Float => operand.kind() == ast::ScalarKind::Bit,
                ast::ScalarKind::Signed => {
                    operand.kind() == ast::ScalarKind::Bit
                        || operand.kind() == ast::ScalarKind::Unsigned
                }
                ast::ScalarKind::Unsigned => {
                    operand.kind() == ast::ScalarKind::Bit
                        || operand.kind() == ast::ScalarKind::Signed
                }
                // A packed half lives in a 32-bit register: the bits are
                // reinterpreted, not converted.
                ast::ScalarKind::Float2 => operand.kind() == ast::ScalarKind::Bit,
                ast::ScalarKind::Pred => false,
            }
        }
        (ast::Type::Vector(inst, _), ast::Type::Vector(operand, _))
        | (ast::Type::Array(inst, _), ast::Type::Array(operand, _)) => {
            should_bitcast(&ast::Type::Scalar(*inst), &ast::Type::Scalar(*operand))
        }
        // A 64-bit value and a pair of 32-bit values hold the same bits; PTX
        // lets a `.b64` register be named as two `.b32` registers
        // (`mov.b64 {%r0,%r1}, %rd2;`).
        (ast::Type::Vector(inst, 2), ast::Type::Scalar(operand)) => {
            inst.size_of() == 4
                && operand.size_of() == 8
                && inst.kind() == ast::ScalarKind::Bit
                && operand.kind() == ast::ScalarKind::Bit
        }
        (ast::Type::Scalar(inst), ast::Type::Vector(operand, 2)) => {
            inst.size_of() == 8
                && operand.size_of() == 4
                && inst.kind() == ast::ScalarKind::Bit
                && operand.kind() == ast::ScalarKind::Bit
        }
        _ => false,
    }
}

fn implicit_conversion_mov(
    (operand_space, operand_type): (ast::StateSpace, &ast::Type),
    (instruction_space, instruction_type): (ast::StateSpace, &ast::Type),
) -> Result<Option<ConversionKind>, TranslateError> {
    // instruction_space is always reg
    if operand_space.is_compatible(ast::StateSpace::Reg) {
        if let (ast::Type::Vector(vec_underlying_type, vec_len), ast::Type::Scalar(scalar)) =
            (operand_type, instruction_type)
        {
            if scalar.kind() == ast::ScalarKind::Bit
                && scalar.size_of() == (vec_underlying_type.size_of() * vec_len)
            {
                return Ok(Some(ConversionKind::Default));
            }
        }
    // Param is addressable only because clang-emitted PTX materializes
    // by-value struct params (.param vectors/arrays) through mov.b64
    // address-of. TODO: verify the kernel-arg, func-arg and variable cases.
    } else if operand_space.is_addressable() || operand_space == ast::StateSpace::Param {
        return Ok(Some(ConversionKind::AddressOf));
    }
    default_implicit_conversion(
        (operand_space, operand_type),
        (instruction_space, instruction_type),
    )
}

fn should_convert_relaxed_src_wrapper(
    (operand_space, operand_type): (ast::StateSpace, &ast::Type),
    (instruction_space, instruction_type): (ast::StateSpace, &ast::Type),
) -> Result<Option<ConversionKind>, TranslateError> {
    if !operand_space.is_compatible(instruction_space) {
        return Err(error_mismatched_type());
    }
    if operand_type == instruction_type {
        return Ok(None);
    }
    match should_convert_relaxed_src(operand_type, instruction_type) {
        conv @ Some(_) => Ok(conv),
        None => Err(error_mismatched_type()),
    }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-size-exceeding-instruction-type-size__relaxed-type-checking-rules-source-operands
fn should_convert_relaxed_src(
    src_type: &ast::Type,
    instr_type: &ast::Type,
) -> Option<ConversionKind> {
    if src_type == instr_type {
        return None;
    }
    match (src_type, instr_type) {
        (ast::Type::Scalar(src_type), ast::Type::Scalar(instr_type)) => match instr_type.kind() {
            ast::ScalarKind::Bit => {
                if instr_type.size_of() <= src_type.size_of() {
                    Some(ConversionKind::Default)
                } else {
                    None
                }
            }
            ast::ScalarKind::Signed | ast::ScalarKind::Unsigned => {
                if instr_type.size_of() <= src_type.size_of()
                    && src_type.kind() != ast::ScalarKind::Float
                {
                    Some(ConversionKind::Default)
                } else {
                    None
                }
            }
            ast::ScalarKind::Float => {
                if instr_type.size_of() <= src_type.size_of()
                    && src_type.kind() == ast::ScalarKind::Bit
                {
                    Some(ConversionKind::Default)
                } else {
                    None
                }
            }
            ast::ScalarKind::Float2 => todo!(),
            ast::ScalarKind::Pred => None,
        },
        (ast::Type::Vector(dst_type, _), ast::Type::Vector(instr_type, _))
        | (ast::Type::Array(dst_type, _), ast::Type::Array(instr_type, _)) => {
            should_convert_relaxed_src(
                &ast::Type::Scalar(*dst_type),
                &ast::Type::Scalar(*instr_type),
            )
        }
        _ => None,
    }
}

fn should_convert_relaxed_dst_wrapper(
    (operand_space, operand_type): (ast::StateSpace, &ast::Type),
    (instruction_space, instruction_type): (ast::StateSpace, &ast::Type),
) -> Result<Option<ConversionKind>, TranslateError> {
    if !operand_space.is_compatible(instruction_space) {
        return Err(error_mismatched_type());
    }
    if operand_type == instruction_type {
        return Ok(None);
    }
    match should_convert_relaxed_dst(operand_type, instruction_type) {
        conv @ Some(_) => Ok(conv),
        None => Err(error_mismatched_type()),
    }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#operand-size-exceeding-instruction-type-size__relaxed-type-checking-rules-destination-operands
fn should_convert_relaxed_dst(
    dst_type: &ast::Type,
    instr_type: &ast::Type,
) -> Option<ConversionKind> {
    if dst_type == instr_type {
        return None;
    }
    match (dst_type, instr_type) {
        (ast::Type::Scalar(dst_type), ast::Type::Scalar(instr_type)) => match instr_type.kind() {
            ast::ScalarKind::Bit => {
                if instr_type.size_of() <= dst_type.size_of() {
                    Some(ConversionKind::Default)
                } else {
                    None
                }
            }
            ast::ScalarKind::Signed => {
                if dst_type.kind() != ast::ScalarKind::Float {
                    if instr_type.size_of() == dst_type.size_of() {
                        Some(ConversionKind::Default)
                    } else if instr_type.size_of() < dst_type.size_of() {
                        Some(ConversionKind::SignExtend)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            ast::ScalarKind::Unsigned => {
                if instr_type.size_of() <= dst_type.size_of()
                    && dst_type.kind() != ast::ScalarKind::Float
                {
                    Some(ConversionKind::Default)
                } else {
                    None
                }
            }
            ast::ScalarKind::Float => {
                if instr_type.size_of() <= dst_type.size_of()
                    && dst_type.kind() == ast::ScalarKind::Bit
                {
                    Some(ConversionKind::Default)
                } else {
                    None
                }
            }
            ast::ScalarKind::Float2 => todo!(),
            ast::ScalarKind::Pred => None,
        },
        (ast::Type::Vector(dst_type, _), ast::Type::Vector(instr_type, _))
        | (ast::Type::Array(dst_type, _), ast::Type::Array(instr_type, _)) => {
            should_convert_relaxed_dst(
                &ast::Type::Scalar(*dst_type),
                &ast::Type::Scalar(*instr_type),
            )
        }
        _ => None,
    }
}

impl<'a> ast::MethodDeclaration<'a, &'a str> {
    fn name(&self) -> &'a str {
        match self.name {
            ast::MethodName::Kernel(name) => name,
            ast::MethodName::Func(name) => name,
        }
    }
}

impl<'a> ast::MethodDeclaration<'a, spirv::Word> {
    fn effective_input_arguments(&self) -> impl Iterator<Item = (spirv::Word, SpirvType)> + '_ {
        let is_kernel = self.name.is_kernel();
        self.input_arguments
            .iter()
            .map(move |arg| {
                if !is_kernel && arg.state_space != ast::StateSpace::Reg {
                    let spirv_type =
                        SpirvType::pointer_to(arg.v_type.clone(), arg.state_space.to_spirv());
                    (arg.name, spirv_type)
                } else {
                    (arg.name, SpirvType::new(arg.v_type.clone()))
                }
            })
            .chain(self.shared_mem.iter().map(|id| {
                (
                    *id,
                    SpirvType::Pointer(
                        Box::new(SpirvType::Base(SpirvScalarKey::B8)),
                        spirv::StorageClass::Workgroup,
                    ),
                )
            }))
    }
}

impl<'input, ID> ast::MethodName<'input, ID> {
    fn is_kernel(&self) -> bool {
        match self {
            ast::MethodName::Kernel(..) => true,
            ast::MethodName::Func(..) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast;

    static SCALAR_TYPES: [ast::ScalarType; 15] = [
        ast::ScalarType::B8,
        ast::ScalarType::B16,
        ast::ScalarType::B32,
        ast::ScalarType::B64,
        ast::ScalarType::S8,
        ast::ScalarType::S16,
        ast::ScalarType::S32,
        ast::ScalarType::S64,
        ast::ScalarType::U8,
        ast::ScalarType::U16,
        ast::ScalarType::U32,
        ast::ScalarType::U64,
        ast::ScalarType::F16,
        ast::ScalarType::F32,
        ast::ScalarType::F64,
    ];

    static RELAXED_SRC_CONVERSION_TABLE: &'static str =
        "b8 	- 	chop 	chop 	chop 	- 	chop 	chop 	chop 	- 	chop 	chop 	chop 	chop 	chop 	chop
        b16 	inv 	- 	chop 	chop 	inv 	- 	chop 	chop 	inv 	- 	chop 	chop 	- 	chop 	chop
        b32 	inv 	inv 	- 	chop 	inv 	inv 	- 	chop 	inv 	inv 	- 	chop 	inv 	- 	chop
        b64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	-
        s8 	- 	chop 	chop 	chop 	- 	chop 	chop 	chop 	- 	chop 	chop 	chop 	inv 	inv 	inv
        s16 	inv 	- 	chop 	chop 	inv 	- 	chop 	chop 	inv 	- 	chop 	chop 	inv 	inv 	inv
        s32 	inv 	inv 	- 	chop 	inv 	inv 	- 	chop 	inv 	inv 	- 	chop 	inv 	inv 	inv
        s64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv
        u8 	- 	chop 	chop 	chop 	- 	chop 	chop 	chop 	- 	chop 	chop 	chop 	inv 	inv 	inv
        u16 	inv 	- 	chop 	chop 	inv 	- 	chop 	chop 	inv 	- 	chop 	chop 	inv 	inv 	inv
        u32 	inv 	inv 	- 	chop 	inv 	inv 	- 	chop 	inv 	inv 	- 	chop 	inv 	inv 	inv
        u64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv
        f16 	inv 	- 	chop 	chop 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	- 	inv 	inv
        f32 	inv 	inv 	- 	chop 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	- 	inv
        f64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	-";

    static RELAXED_DST_CONVERSION_TABLE: &'static str =
        "b8 	- 	zext 	zext 	zext 	- 	zext 	zext 	zext 	- 	zext 	zext 	zext 	zext 	zext 	zext
        b16 	inv 	- 	zext 	zext 	inv 	- 	zext 	zext 	inv 	- 	zext 	zext 	- 	zext 	zext
        b32 	inv 	inv 	- 	zext 	inv 	inv 	- 	zext 	inv 	inv 	- 	zext 	inv 	- 	zext
        b64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	-
        s8 	- 	sext 	sext 	sext 	- 	sext 	sext 	sext 	- 	sext 	sext 	sext 	inv 	inv 	inv
        s16 	inv 	- 	sext 	sext 	inv 	- 	sext 	sext 	inv 	- 	sext 	sext 	inv 	inv 	inv
        s32 	inv 	inv 	- 	sext 	inv 	inv 	- 	sext 	inv 	inv 	- 	sext 	inv 	inv 	inv
        s64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv
        u8 	- 	zext 	zext 	zext 	- 	zext 	zext 	zext 	- 	zext 	zext 	zext 	inv 	inv 	inv
        u16 	inv 	- 	zext 	zext 	inv 	- 	zext 	zext 	inv 	- 	zext 	zext 	inv 	inv 	inv
        u32 	inv 	inv 	- 	zext 	inv 	inv 	- 	zext 	inv 	inv 	- 	zext 	inv 	inv 	inv
        u64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv 	- 	inv 	inv 	inv
        f16 	inv 	- 	zext 	zext 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	- 	inv 	inv
        f32 	inv 	inv 	- 	zext 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	- 	inv
        f64 	inv 	inv 	inv 	- 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	inv 	-";

    fn table_entry_to_conversion(entry: &'static str) -> Option<ConversionKind> {
        match entry {
            "-" => Some(ConversionKind::Default),
            "inv" => None,
            "zext" => Some(ConversionKind::Default),
            "chop" => Some(ConversionKind::Default),
            "sext" => Some(ConversionKind::SignExtend),
            _ => unreachable!(),
        }
    }

    fn parse_conversion_table(table: &'static str) -> Vec<Vec<Option<ConversionKind>>> {
        table
            .lines()
            .map(|line| {
                line.split_ascii_whitespace()
                    .skip(1)
                    .map(table_entry_to_conversion)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn assert_conversion_table<F: Fn(&ast::Type, &ast::Type) -> Option<ConversionKind>>(
        table: &'static str,
        f: F,
    ) {
        let conv_table = parse_conversion_table(table);
        for (instr_idx, instr_type) in SCALAR_TYPES.iter().enumerate() {
            for (op_idx, op_type) in SCALAR_TYPES.iter().enumerate() {
                let conversion = f(
                    &ast::Type::Scalar(*op_type),
                    &ast::Type::Scalar(*instr_type),
                );
                if instr_idx == op_idx {
                    assert!(conversion == None);
                } else {
                    assert!(conversion == conv_table[instr_idx][op_idx]);
                }
            }
        }
    }

    #[test]
    fn should_convert_relaxed_src_all_combinations() {
        assert_conversion_table(RELAXED_SRC_CONVERSION_TABLE, should_convert_relaxed_src);
    }

    #[test]
    fn should_convert_relaxed_dst_all_combinations() {
        assert_conversion_table(RELAXED_DST_CONVERSION_TABLE, should_convert_relaxed_dst);
    }
}

/// Human-readable type name, for error messages.
fn describe_type(ty: &ast::Type) -> String {
    match ty {
        ast::Type::Scalar(s) => format!("scalar({})", scalar_name(s)),
        ast::Type::Vector(s, n) => format!("vector({},{})", scalar_name(s), n),
        ast::Type::Array(s, n) => format!("array({},{:?})", scalar_name(s), n),
        ast::Type::Pointer(t, _) => format!("pointer({})", scalar_name(t)),
        ast::Type::Struct(members) => {
            format!("struct({} members)", members.len())
        }
    }
}

fn scalar_name(s: &ast::ScalarType) -> String {
    match s {
        ast::ScalarType::B8 => "b8".to_string(),
        ast::ScalarType::B16 => "b16".to_string(),
        ast::ScalarType::B32 => "b32".to_string(),
        ast::ScalarType::B64 => "b64".to_string(),
        ast::ScalarType::U8 => "u8".to_string(),
        ast::ScalarType::U16 => "u16".to_string(),
        ast::ScalarType::U32 => "u32".to_string(),
        ast::ScalarType::U64 => "u64".to_string(),
        ast::ScalarType::S8 => "s8".to_string(),
        ast::ScalarType::S16 => "s16".to_string(),
        ast::ScalarType::S32 => "s32".to_string(),
        ast::ScalarType::S64 => "s64".to_string(),
        ast::ScalarType::F16 => "f16".to_string(),
        ast::ScalarType::F16x2 => "f16x2".to_string(),
        ast::ScalarType::F32 => "f32".to_string(),
        ast::ScalarType::F64 => "f64".to_string(),
        ast::ScalarType::Pred => "pred".to_string(),
    }
}

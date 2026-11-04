//! CUDA 12 "library" API (`cuLibraryLoadData` and friends).
//!
//! libcudart 12.4 resolves the module loader with plain `dlsym` on
//! `libcuda.so.1` rather than through the `CUDART_INTERFACE` dark API, and
//! falls back to an internal stub failing with `cudaErrorNotSupported` when the
//! symbol is missing. These entry points therefore have to exist for the
//! runtime API to work at all.
//!
//! A `CUlibrary` is a loaded module; a `CUkernel` is a function inside one.

use super::module::{self, Module};
use super::{CUresult, Decuda, Encuda, GlobalState};
use crate::cuda;
use std::ffi::{c_char, CStr};
use std::os::raw::{c_uint, c_void};
use std::ptr;

/// Opaque `CUlibrary` handle. Backed by a `Module`.
pub struct Library {
    module: *mut Module,
}

/// Opaque `CUkernel` handle. Backed by a `Function`.
pub struct Kernel {
    function: *mut super::function::Function,
}

impl Library {
    fn into_raw(self) -> *mut Library {
        Box::into_raw(Box::new(self))
    }
}

impl Kernel {
    fn into_raw(self) -> *mut Kernel {
        Box::into_raw(Box::new(self))
    }
}

/// `cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions,
///                     libraryOptions, libraryOptionValues, numLibraryOptions)`
///
/// Loads a fatbin (or plain PTX) into a new library handle.
pub(crate) fn load_data(
    library: *mut *mut Library,
    code: *const c_void,
    _jit_options: *mut crate::cuda::CUjit_option,
    _jit_options_values: *mut *mut c_void,
    _num_jit_options: c_uint,
    _library_options: *mut c_uint,
    _library_option_values: *mut *mut c_void,
    _num_library_options: c_uint,
) -> Result<(), CUresult> {
    if library.is_null() || code.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut module: *mut Module = ptr::null_mut();
    module::load_fatbin(&mut module, code)?;
    let lib = Library { module };
    unsafe { *library = lib.into_raw() };
    Ok(())
}

/// `cuLibraryLoadFromFile(library, fileName, ...)`
pub(crate) fn load_from_file(
    library: *mut *mut Library,
    file_name: *const c_char,
    _jit_options: *mut crate::cuda::CUjit_option,
    _jit_options_values: *mut *mut c_void,
    _num_jit_options: c_uint,
    _library_options: *mut c_uint,
    _library_option_values: *mut *mut c_void,
    _num_library_options: c_uint,
) -> Result<(), CUresult> {
    if library.is_null() || file_name.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let path = unsafe { std::ffi::CStr::from_ptr(file_name) }.to_string_lossy().into_owned();
    let bytes = std::fs::read(&path).map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
    let mut module: *mut Module = ptr::null_mut();
    module::load_data(&mut module, bytes.as_ptr() as *const c_void)?;
    let lib = Library { module };
    unsafe { *library = lib.into_raw() };
    Ok(())
}

/// `cuLibraryUnload(library)`
pub(crate) fn unload(library: *mut Library) -> Result<(), CUresult> {
    if library.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let lib = unsafe { Box::from_raw(library) };
    if !lib.module.is_null() {
        module::unload(lib.module)?;
    }
    Ok(())
}

/// `cuLibraryGetKernel(kernel, library, name)`
pub(crate) fn get_kernel(
    kernel: *mut *mut c_void,
    library: *mut Library,
    name: *const c_char,
) -> Result<(), CUresult> {
    if kernel.is_null() || library.is_null() || name.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let lib = unsafe { &*library };
    if lib.module.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    let mut function: *mut super::function::Function = ptr::null_mut();
    if let Err(e) = module::get_function(&mut function, lib.module, name) {
        if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
            eprintln!(
                "cuLibraryGetKernel({:?}) failed: {:?}",
                unsafe { CStr::from_ptr(name) }.to_string_lossy(),
                e
            );
        }
        return Err(e);
    }
    // The runtime hands this handle back to the driver as a CUfunction (for
    // cuFuncGetAttribute and cuLaunchKernelEx), so it has to be the function
    // handle itself, not a wrapper around it.
    if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
        eprintln!(
            "cuLibraryGetKernel({:?}) -> {:?}",
            unsafe { CStr::from_ptr(name) }.to_string_lossy(),
            function
        );
    }
    unsafe { *kernel = function as *mut c_void };
    Ok(())
}

/// `cuLibraryGetModule(module, library)`
pub(crate) fn get_module(
    module_out: *mut crate::cuda::CUmodule,
    library: *mut Library,
) -> Result<(), CUresult> {
    if module_out.is_null() || library.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let lib = unsafe { &*library };
    if lib.module.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    unsafe { *module_out = lib.module as *mut crate::cuda::CUmod_st };
    Ok(())
}

/// `cuLibraryGetGlobal(global, library, name, size)`
///
/// Globals are not tracked by the module loader, so there is nothing to hand
/// back.
pub(crate) fn get_global(
    global_out: *mut crate::cuda::CUdeviceptr,
    library: *mut Library,
    _name: *const c_char,
    _size: *mut usize,
) -> Result<(), CUresult> {
    if global_out.is_null() || library.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let lib = unsafe { &*library };
    if lib.module.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    Err(CUresult::CUDA_ERROR_NOT_FOUND)
}

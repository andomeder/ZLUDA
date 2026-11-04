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
    pub arg_size: Vec<usize>,
    pub use_shared_mem: bool,
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
    GlobalState::lock_enqueue(hstream, |cmd_list, signal, wait| {
        let func: &mut FunctionData = match unsafe { &mut *f }.as_result_mut() {
            Ok(func) => func,
            Err(e) => {
                if trace {
                    eprintln!("launch_kernel: handle {:?} is not a live function: {:?}", f, e);
                }
                return Err(e);
            }
        };
        if kernel_params != ptr::null_mut() {
            for (i, arg_size) in func.arg_size.iter().enumerate() {
                if let Err(e) = unsafe {
                    func.base
                        .set_arg_raw(i as u32, *arg_size, *kernel_params.add(i))
                } {
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
            unsafe {
                func.base.set_arg_raw(
                    func.arg_size.len() as u32,
                    shared_mem_bytes as usize,
                    ptr::null(),
                )?
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
    // ZLUDA_SYNC_LAUNCH makes every launch block until it retires. It is a
    // diagnostic: a difference between running with it and without it is the
    // difference between a missing ordering and a wrong value.
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
            // Attributes ZLUDA does not model (the cluster scheduling ones,
            // for instance) read as zero, the way CUDA reports the attributes
            // that do not apply to a kernel. Returning "not supported" here
            // breaks cudaFuncGetAttributes, which fills a whole struct.
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
            // The L0 driver sizes dynamic shared memory per-launch via the
            // sharedMemBytes argument to cuLaunchKernel, so the setter has no
            // per-function state to persist.
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            | CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT => {
                Ok(())
            }
            // Unknown/advisory attributes (cluster dims, cache prefs,
            // scheduling policy...) are safe to accept-and-ignore on our
            // single-die in-order queues; rejecting them breaks runtimes
            // that probe-set them unconditionally (e.g. MAX).
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

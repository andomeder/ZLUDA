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
    /// What the kernel can actually address, when its PTX proves it: the
    /// declared prefix above is the size of the symbol, which is the module's
    /// static shared size and can be many times what the kernel touches.
    pub shared_bound: Option<super::shared_window::SharedBound>,
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
    let timing = super::diag::trace_launch_ms();
    let trace = super::launch_trace::Trace::new(timing);
    // The whole call, including the diagnostic environment lookups below, so
    // that what the phases do not account for is visible rather than hidden.
    let t_enter = std::time::Instant::now();
    let trace_func = super::diag::trace_func();
    if f == ptr::null_mut()
        || (kernel_params == ptr::null_mut() && extra == ptr::null_mut())
        || (kernel_params != ptr::null_mut() && extra != ptr::null_mut())
    {
        if trace_func {
            eprintln!(
                "launch_kernel: bad arguments f={:?} kernel_params={:?} extra={:?}",
                f, kernel_params, extra
            );
        }
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    if trace_func {
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
    let shared_trace = super::diag::trace_shared();
    // A kernel whose arguments do not fit in the parameter list reads them from
    // a buffer. Gathering the bytes is plain host work; filling a buffer here
    // rather than inside the lock is deliberate, because a device allocation
    // takes the global state lock and the launch path holds the stream.
    let mut param_block = None;
    let mut param_bytes = None;
    let function_key = f as usize;
    let stream_key = hstream as usize;
    // ZLUDA_PARAM_BLOCKING=1 restores the old path: allocate a buffer per launch
    // and fill it with a copy that drains the context. Slow (2.5 ms per launch on
    // torch's cat) but kept as the reference the staged path is measured against.
    let blocking = super::diag::param_blocking();
    let t_gather = trace.start();
    if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
        if let Some(layout) = func.param_buffer.clone() {
            if blocking {
                param_block = Some(super::param_buffer::publish(&layout, kernel_params, extra)?);
            } else {
                super::param_buffer::reserve(&layout, function_key, stream_key).map_err(|error| {
                    eprintln!("launch_kernel: reserve failed {:?}", error);
                    error
                })?;
                let bytes = super::param_buffer::collect(&layout, kernel_params, extra)?;
                if super::param_buffer::in_device() {
                    // Room first, then fill: a slot is only overwritten once the
                    // launch that read it retired. The copy goes on the launch's
                    // own stream, ahead of the kernel, so it is ordered.
                    GlobalState::wait_stream_pending(hstream, super::param_buffer::slots())?;
                    let (device, host) =
                        super::param_buffer::take_slot(&layout, &bytes, function_key, stream_key)?;
                    super::memory::copy_htod_async(
                        device as *mut std::ffi::c_void,
                        host as *const std::ffi::c_void,
                        layout.total,
                        hstream,
                    )?;
                    param_block = Some(device as *mut std::ffi::c_void);
                } else {
                    param_bytes = Some((layout.clone(), bytes));
                }
            }
        }
    }
    // A launch with no parameter buffer has nothing to wait for.
    let keep_pending = if param_bytes.is_some() {
        super::param_buffer::slots()
    } else {
        usize::MAX
    };
    trace.add(t_gather, |t| &t.gather);
    let wait_last = param_block.is_some();
    GlobalState::lock_enqueue_staged(hstream, keep_pending, wait_last, &trace, |cmd_list, signal, wait| {
        let func: &mut FunctionData = match unsafe { &mut *f }.as_result_mut() {
            Ok(func) => func,
            Err(e) => {
                if trace_func {
                    eprintln!("launch_kernel: handle {:?} is not a live function: {:?}", f, e);
                }
                return Err(e);
            }
        };
        // ZLUDA_DUMP_PARAMS=<substring> prints the bytes of every argument of a
        // matching launch. A kernel that only misbehaves for some inputs cannot
        // be diagnosed from the source alone, and the parameter block is the
        // only place the difference shows.
        let dump_this = super::diag::dump_params(&func.name.to_string_lossy());
        let traced_stage = super::diag::trace_stage();
        if let Some((layout, bytes)) = param_bytes.take() {
            // The slot pool rotates, so this launch gets a buffer that no
            // launch still in flight can be reading. Its copy is appended to
            // this list, ahead of the launch, which is all the ordering a
            // single queue needs.
            let t_stage_start = std::time::Instant::now();
            match super::param_buffer::stage(&layout, &bytes, function_key, stream_key, cmd_list) {
                Ok(block) => {
                    if traced_stage {
                        eprintln!("launch: staged block {:p} for {}", block, func.name.to_string_lossy());
                    }
                    param_block = Some(block);
                    let _ = &t_stage_start;
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
            let t = trace.start();
            unsafe { func.base.set_arg_raw(0, std::mem::size_of::<*mut c_void>(), &block as *const _ as *const c_void)? };
            trace.add(t, |t| &t.set_arg);
        } else if kernel_params != ptr::null_mut() {
            let t = trace.start();
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
                    if trace_func {
                        eprintln!(
                            "launch_kernel: set_arg_raw({}, {}) failed: {:?}",
                            i, arg_size, e
                        );
                    }
                    return Err(e.into());
                }
            }
            trace.add(t, |t| &t.set_arg);
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
                        if trace_func {
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
                        if trace_func {
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
            // prefix the kernel assumes sits in front of it. The declared
            // prefix is the size of the shared symbol, which for a lifted
            // kernel is the whole module's static shared size, so it can be
            // many times what the kernel addresses. When the PTX proves a
            // bound, that bound is what decides - never below the
            // application's own request, never above the declared window.
            let declared_window = shared_mem_bytes as usize + func.shared_mem_prefix;
            let window = match func.shared_bound {
                Some(bound) => {
                    let threads = block_dim_x as u64 * block_dim_y as u64 * block_dim_z as u64;
                    let needed = bound.bytes(threads) as usize;
                    declared_window.min(needed.max(shared_mem_bytes as usize))
                }
                None => declared_window,
            };
            if shared_trace {
                eprintln!(
                    "shared: {} prefix={} app={} total={} declared={} bound={:?}{}",
                    func.name.to_string_lossy(),
                    func.shared_mem_prefix,
                    shared_mem_bytes,
                    window,
                    declared_window,
                    func.shared_bound,
                    if window < declared_window { " narrowed" } else { "" }
                );
            }
            let t = trace.start();
            unsafe {
                func.base.set_arg_raw(own_parameters as u32, window, ptr::null())?
            };
            trace.add(t, |t| &t.shared);
        } else if shared_trace {
            eprintln!(
                "shared: {} prefix=0 app={} total={} (no window)",
                func.name.to_string_lossy(),
                shared_mem_bytes,
                shared_mem_bytes
            );
        }
        let t = trace.start();
        if let Err(e) = func.base.set_group_size(block_dim_x, block_dim_y, block_dim_z) {
            if trace.enabled {
                eprintln!(
                    "launch_kernel: set_group_size({}, {}, {}) failed: {:?}",
                    block_dim_x, block_dim_y, block_dim_z, e
                );
            }
            return Err(e.into());
        }
        trace.add(t, |t| &t.group);
        func.legacy_args.reset();
        let t_append = trace.start();
        unsafe {
            // Launching without this wait is tempting and wrong: the list is in
            // order, but a decode step's launches stop completing altogether
            // without it, so something in the chain is load bearing. What it
            // costs is measured though - a step's 2167 launches take 212 ms
            // queued against 40 ms when each is run alone, so ~80 us per launch
            // goes on the event round trip. Narrowing which launches need the
            // wait is the open item.
            let no_wait: &[&l0::Event<'static>] = &[];
            let launch_wait = if super::diag::no_launch_wait() {
                no_wait
            } else {
                wait
            };
            if let Err(e) = cmd_list.append_launch_kernel(
                &mut func.base,
                &[grid_dim_x, grid_dim_y, grid_dim_z],
                Some(signal),
                launch_wait,
            ) {
                if trace_func {
                    eprintln!(
                        "launch_kernel: append_launch_kernel grid ({}, {}, {}) failed: {:?}",
                        grid_dim_x, grid_dim_y, grid_dim_z, e
                    );
                }
                return Err(e.into());
            }
        }
        trace.add(t_append, |t| &t.append);
        Ok::<_, CUresult>(())
    })
    .map_err(|error| {
        // A launch that fails has to say so: the runtime reports the sticky
        // error much later, at a sync, with no idea which kernel caused it.
        if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
            eprintln!(
                "launch_kernel: {} failed {:?}",
                func.name.to_string_lossy(),
                error
            );
        } else {
            eprintln!("launch_kernel: failed {:?}", error);
        }
        error
    })?;
    // ZLUDA_SYNC_LAUNCH makes every launch block until it retires, a
    // diagnostic: a difference with it on separates a missing ordering from a
    // wrong value.
    if super::diag::trace_stream() {
        if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
            super::stream::remember_launched_kernel(&func.name.to_string_lossy());
        }
    }
    if super::diag::sync_launch() {
        crate::r#impl::stream::synchronize(hstream)?;
    }
    if timing {
        // Printed last, so with ZLUDA_SYNC_LAUNCH the number is the kernel's own
        // time: the sync above does not return until it retired.
        if let Ok(func) = unsafe { &mut *f }.as_result_mut() {
            trace.print(
                &func.name.to_string_lossy(),
                t_enter.elapsed().as_secs_f64() * 1e3,
            );
        }
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

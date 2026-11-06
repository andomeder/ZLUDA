use crate::{
    cuda::{
        CUctx_st, CUdevice, CUdeviceptr, CUevent_st, CUfunc_st, CUkern_st, CUlib_st, CUmod_st,
        CUresult, CUstream_st,
    },
    r#impl::device::Device,
};
use std::{
    ffi::c_void,
    mem::{self, ManuallyDrop},
    os::raw::c_int,
    ptr,
    sync::Mutex,
    sync::TryLockError,
};

#[cfg(test)]
#[macro_use]
pub mod test;
pub mod context;
pub mod device;
pub mod diag;
pub mod export_table;
pub mod function;
pub mod library;
pub mod memory;
pub mod module;
pub(crate) mod occupancy;
mod param_buffer;
mod ptx_split;
#[cfg_attr(windows, path = "os_win.rs")]
#[cfg_attr(not(windows), path = "os_unix.rs")]
pub(crate) mod os;
pub mod stream;
pub mod driver_proc_address;
pub mod event;
pub mod fatbin;
pub mod launch_trace;
pub mod pointer;
pub mod sass;

/// An entry point that is known but not implemented.
///
/// Always a returned error, never a panic: this crosses the C ABI, and a panic
/// through that boundary is undefined behaviour on the caller's side - which in
/// practice means a crash in whichever library made the call, with no clue which
/// entry point it was. In a debug build it also names the call.
pub fn unimplemented() -> CUresult {
    if cfg!(debug_assertions) {
        eprintln!(
            "zluda: an unimplemented driver entry point was called (debug build); returning \
             CUDA_ERROR_NOT_SUPPORTED. In a release build, ZLUDA_TRACE_STUBS=1 logs this."
        );
    } else if std::env::var_os("ZLUDA_TRACE_STUBS").is_some() {
        eprintln!("zluda: an unimplemented driver entry point was called, returning CUDA_ERROR_NOT_SUPPORTED");
    }
    CUresult::CUDA_ERROR_NOT_SUPPORTED
}


pub trait HasLivenessCookie: Sized {
    const COOKIE: usize;
    const LIVENESS_FAIL: CUresult;

    fn try_drop(&mut self) -> Result<(), CUresult>;
}

// This struct is a best-effort check if wrapped value has been dropped,
// while it's inherently safe, its use coming from FFI is very unsafe
#[repr(C)]
pub struct LiveCheck<T: HasLivenessCookie> {
    cookie: usize,
    data: ManuallyDrop<T>,
}

impl<T: HasLivenessCookie> LiveCheck<T> {
    pub fn new(data: T) -> Self {
        LiveCheck {
            cookie: T::COOKIE,
            data: ManuallyDrop::new(data),
        }
    }

    fn destroy_impl(this: *mut Self) -> Result<(), CUresult> {
        let mut ctx_box = ManuallyDrop::new(unsafe { Box::from_raw(this) });
        ctx_box.try_drop()?;
        unsafe { ManuallyDrop::drop(&mut ctx_box) };
        Ok(())
    }

    unsafe fn ptr_from_inner(this: *mut T) -> *mut Self {
        let outer_ptr = (this as *mut u8).sub(mem::size_of::<usize>());
        outer_ptr as *mut Self
    }

    pub unsafe fn as_ref_unchecked(&self) -> &T {
        &self.data
    }

    pub fn as_option_mut(&mut self) -> Option<&mut T> {
        if self.cookie == T::COOKIE {
            Some(&mut self.data)
        } else {
            None
        }
    }

    pub fn as_result(&self) -> Result<&T, CUresult> {
        if self.cookie == T::COOKIE {
            Ok(&self.data)
        } else {
            Err(T::LIVENESS_FAIL)
        }
    }

    pub fn as_result_mut(&mut self) -> Result<&mut T, CUresult> {
        if self.cookie == T::COOKIE {
            Ok(&mut self.data)
        } else {
            Err(T::LIVENESS_FAIL)
        }
    }

    #[must_use]
    pub fn try_drop(&mut self) -> Result<(), CUresult> {
        if self.cookie == T::COOKIE {
            self.cookie = 0;
            self.data.try_drop()?;
            unsafe { ManuallyDrop::drop(&mut self.data) };
            return Ok(());
        }
        Err(T::LIVENESS_FAIL)
    }
}

impl<T: HasLivenessCookie> Drop for LiveCheck<T> {
    fn drop(&mut self) {
        self.cookie = 0;
    }
}

pub trait CudaRepr: Sized {
    type Impl: Sized;
}

impl<T: CudaRepr> CudaRepr for *mut T {
    type Impl = *mut T::Impl;
}

pub trait Decuda<To> {
    fn decuda(self: Self) -> To;
}

impl<T: CudaRepr> Decuda<*mut T::Impl> for *mut T {
    fn decuda(self: Self) -> *mut T::Impl {
        self as *mut _
    }
}

impl From<l0::sys::ze_result_t> for CUresult {
    fn from(result: l0::sys::ze_result_t) -> Self {
        if result != l0::sys::ze_result_t::ZE_RESULT_SUCCESS
            && std::env::var_os("ZLUDA_TRACE_L0").is_some()
        {
            eprintln!("level zero returned {}", result.0);
        }
        match result {
            l0::sys::ze_result_t::ZE_RESULT_SUCCESS => CUresult::CUDA_SUCCESS,
            l0_sys::ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED => {
                CUresult::CUDA_ERROR_NOT_INITIALIZED
            }
            l0_sys::ze_result_t::ZE_RESULT_ERROR_INVALID_ENUMERATION
            | l0_sys::ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT
            | l0_sys::ze_result_t::ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION
            | l0_sys::ze_result_t::ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION => {
                CUresult::CUDA_ERROR_INVALID_VALUE
            }
            l0_sys::ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY => {
                CUresult::CUDA_ERROR_OUT_OF_MEMORY
            }
            l0_sys::ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE => {
                CUresult::CUDA_ERROR_NOT_SUPPORTED
            }
            _ => CUresult::CUDA_ERROR_UNKNOWN,
        }
    }
}

impl<T> From<TryLockError<T>> for CUresult {
    fn from(_: TryLockError<T>) -> Self {
        CUresult::CUDA_ERROR_ILLEGAL_STATE
    }
}

pub trait Encuda {
    type To: Sized;
    fn encuda(self: Self) -> Self::To;
}

impl Encuda for CUresult {
    type To = CUresult;
    fn encuda(self: Self) -> Self::To {
        self
    }
}

impl Encuda for l0::sys::ze_result_t {
    type To = CUresult;
    fn encuda(self: Self) -> Self::To {
        self.into()
    }
}

impl Encuda for () {
    type To = CUresult;
    fn encuda(self: Self) -> Self::To {
        CUresult::CUDA_SUCCESS
    }
}

impl<T1: Encuda<To = CUresult>, T2: Encuda<To = CUresult>> Encuda for Result<T1, T2> {
    type To = CUresult;
    fn encuda(self: Self) -> Self::To {
        match self {
            Ok(e) => e.encuda(),
            Err(e) => e.encuda(),
        }
    }
}

lazy_static! {
    static ref GLOBAL_STATE: Mutex<Option<GlobalState>> = Mutex::new(None);
}

struct GlobalState {
    devices: Vec<Device>,
    global_heap: *mut c_void,
}

unsafe impl Send for GlobalState {}

impl GlobalState {
    fn lock<T>(f: impl FnOnce(&mut GlobalState) -> T) -> Result<T, CUresult> {
        let mut mutex = GLOBAL_STATE
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let global_state = mutex.as_mut().ok_or(CUresult::CUDA_ERROR_ILLEGAL_STATE)?;
        Ok(f(global_state))
    }

    fn lock_device<T>(
        device::Index(dev_idx): device::Index,
        f: impl FnOnce(&'static mut device::Device) -> T,
    ) -> Result<T, CUresult> {
        if dev_idx < 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_DEVICE);
        }
        Self::lock(|global_state| {
            if dev_idx >= global_state.devices.len() as c_int {
                Err(CUresult::CUDA_ERROR_INVALID_DEVICE)
            } else {
                Ok(f(unsafe {
                    transmute_lifetime_mut(&mut global_state.devices[dev_idx as usize])
                }))
            }
        })?
    }

    fn lock_current_context<F: FnOnce(&mut context::ContextData) -> R, R>(
        f: F,
    ) -> Result<R, CUresult> {
        Self::lock_current_context_unchecked(|ctx| Ok(f(ctx.as_result_mut()?)))?
    }

    fn lock_current_context_unchecked<F: FnOnce(&mut context::Context) -> R, R>(
        f: F,
    ) -> Result<R, CUresult> {
        context::CONTEXT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let ctx = match stack.last_mut() {
                Some(ctx) => ctx,
                None => {
                    // Thread has no pushed context (worker thread / child after
                    // fork). Fall back to the process-global primary context so
                    // cuMemFree/cuMemAlloc etc. still work, matching real CUDA's
                    // implicit-primary-context behavior.
                    let primary = context::get_primary_context();
                    if primary.is_null() {
                        return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
                    }
                    return GlobalState::lock(|_| f(unsafe { &mut *primary }));
                }
            };
            GlobalState::lock(|_| f(unsafe { &mut **ctx }))
        })
    }

    fn lock_stream<T>(
        stream: *mut stream::Stream,
        f: impl FnOnce(&mut stream::StreamData) -> T,
    ) -> Result<T, CUresult> {
        if stream == ptr::null_mut()
            || stream == stream::CU_STREAM_LEGACY
            || stream == stream::CU_STREAM_PER_THREAD
        {
            Self::lock_current_context(|ctx| Ok(f(&mut ctx.default_stream)))?
        } else {
            Self::lock(|_| {
                let stream = unsafe { &mut *stream }.as_result_mut()?;
                Ok(f(stream))
            })?
        }
    }

    /// Drain every stream of the current context (default + all created).
    /// Safe to call while holding the global lock (StreamData::synchronize
    /// takes no locks of its own); must NOT be called with another
    /// lock_current_context/lock_stream closure active on this thread.
    pub(crate) fn synchronize_context_streams() -> Result<(), CUresult> {
        Self::lock_current_context(|ctx| {
            // A drain that timed out has to be reported: its callers unmap
            // memory afterwards, and unmapping memory that a queued kernel
            // still references resets the execution queue.
            ctx.default_stream.synchronize()?;
            for s in ctx.streams.iter() {
                let stream_data = unsafe { &mut **s };
                stream_data.synchronize()?;
            }
            Ok::<_, CUresult>(())
        })??;
        Ok(())
    }

    fn lock_enqueue(
        stream: *mut stream::Stream,
        f: impl FnOnce(
            &l0::CommandList,
            &l0::Event<'static>,
            &[&l0::Event<'static>],
        ) -> Result<(), CUresult>,
    ) -> Result<(), CUresult> {
        Self::lock_stream(stream, |stream_data| {
            let l0_dev = unsafe { (*(*stream_data.context).device).base };
            let l0_ctx = unsafe { &mut (*(*stream_data.context).device).l0_context };
            let cmd_list = unsafe { transmute_lifetime(&stream_data.cmd_list) };
            // TODO: make new_marker drop-safe
            let (new_event, new_marker) = stream_data.get_event(l0_dev, l0_ctx)?;
            stream_data.try_reuse_finished_events()?;
            let prev_event = stream_data.get_last_event();
            let prev_event_array = prev_event.map(|e| [e]);
            let empty = [];
            let prev_event_slice = prev_event_array.as_ref().map_or(&empty[..], |arr| &arr[..]);
            f(cmd_list, &new_event, prev_event_slice)?;
            stream_data.push_event((new_event, new_marker));
            Ok(())
        })?
    }

    /// The same enqueue as `lock_enqueue`, but first waits until at most
    /// `keep_pending` launches are still in flight.
    ///
    /// A launch that fills a parameter buffer needs that: it reuses a buffer an
    /// earlier launch read, and the stream's own markers are what say that
    /// launch has retired. The wait happens here, before the previous marker is
    /// borrowed, because retiring markers mutates the queue.
    /// Wait until at most `keep` launches are still in flight on a stream.
    fn wait_stream_pending(stream: *mut stream::Stream, keep: usize) -> Result<(), CUresult> {
        Self::lock_stream(stream, |stream_data| {
            stream_data.wait_for_pending(keep).map_err(CUresult::from)
        })?
    }

    fn lock_enqueue_staged(
        stream: *mut stream::Stream,
        keep_pending: usize,
        wait_last: bool,
        trace: &launch_trace::Trace,
        f: impl FnOnce(
            &l0::CommandList,
            &l0::Event<'static>,
            &[&l0::Event<'static>],
        ) -> Result<(), CUresult>,
    ) -> Result<(), CUresult> {
        Self::lock_stream(stream, |stream_data| {
            let l0_dev = unsafe { (*(*stream_data.context).device).base };
            let l0_ctx = unsafe { &mut (*(*stream_data.context).device).l0_context };
            let cmd_list = unsafe { transmute_lifetime(&stream_data.cmd_list) };
            let t_wait = trace.start();
            stream_data.wait_for_pending(keep_pending)?;
            trace.add(t_wait, |t| &t.lock);
            let t1 = trace.start();
            let (new_event, new_marker) = stream_data.get_event(l0_dev, l0_ctx)?;
            trace.add(t1, |t| &t.event);
            let t2 = trace.start();
            stream_data.try_reuse_finished_events()?;
            trace.add(t2, |t| &t.reuse);
            // Which earlier command a new one waits on. One means the command
            // before it, and that round trip measured about 80 us per launch -
            // a decode step of 2167 launches spent 212 ms queued against 40 ms
            // with each launch run alone. Waiting `lag` commands back keeps a
            // dependency chain (removing it entirely deadlocks the runtime) but
            // lets that many commands be in flight instead of one.
            // A launch whose arguments were just copied into a buffer has to
            // wait for that copy, which is the most recent command, so it takes
            // the immediate predecessor regardless of the lag.
            let lag = if wait_last { 1 } else { diag::wait_lag() };
            let prev_event = if lag <= 1 {
                stream_data.get_last_event()
            } else {
                let len = stream_data.busy_events.len();
                if len >= lag {
                    stream_data.busy_events.get(len - lag).map(|(event, _)| event)
                } else {
                    None
                }
            };
            let prev_event_array = prev_event.map(|e| [e]);
            let empty = [];
            let prev_event_slice = prev_event_array.as_ref().map_or(&empty[..], |arr| &arr[..]);
            f(cmd_list, &new_event, prev_event_slice)?;
            let t_book = trace.start();
            stream_data.push_event((new_event, new_marker));
            trace.add(t_book, |t| &t.book);
            Ok(())
        })?
    }

    fn lock_function<T>(
        func: *mut function::Function,
        f: impl FnOnce(&mut function::FunctionData) -> T,
    ) -> Result<T, CUresult> {
        if func == ptr::null_mut() {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        Self::lock(|_| {
            let func = unsafe { &mut *func }.as_result_mut()?;
            Ok(f(func))
        })?
    }
}

// TODO: implement
fn is_intel_gpu_driver(_: &l0::Driver) -> bool {
    true
}

pub fn init() -> Result<(), CUresult> {
    let mut global_state = GLOBAL_STATE
        .lock()
        .map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
    if global_state.is_some() {
        return Ok(());
    }
    l0::init()?;
    let drivers = l0::Driver::get()?;
    let devices = match drivers.into_iter().find(is_intel_gpu_driver) {
        None => return Err(CUresult::CUDA_ERROR_UNKNOWN),
        Some(driver) => device::init(&driver)?,
    };
    let global_heap = unsafe { os::heap_create() };
    if global_heap == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_OUT_OF_MEMORY);
    }
    *global_state = Some(GlobalState {
        devices,
        global_heap,
    });
    drop(global_state);
    Ok(())
}

macro_rules! stringify_curesult {
    ($x:ident => [ $($variant:ident),+ ]) => {
        match $x {
            $(
                CUresult::$variant => Some(concat!(stringify!($variant), "\0")),
            )+
            _ => None
        }
    }
}

pub(crate) fn get_error_name(error: CUresult, str: *mut *const i8) -> CUresult {
    if str == ptr::null_mut() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let text = match error {
        CUresult::CUDA_SUCCESS => "CUDA_SUCCESS",
        CUresult::CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
        CUresult::CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY",
        CUresult::CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
        CUresult::CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE",
        CUresult::CUDA_ERROR_INVALID_IMAGE => "CUDA_ERROR_INVALID_IMAGE",
        CUresult::CUDA_ERROR_INVALID_CONTEXT => "CUDA_ERROR_INVALID_CONTEXT",
        CUresult::CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND",
        CUresult::CUDA_ERROR_NOT_READY => "CUDA_ERROR_NOT_READY",
        CUresult::CUDA_ERROR_LAUNCH_FAILED => "CUDA_ERROR_LAUNCH_FAILED",
        CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => {
            "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"
        }
        CUresult::CUDA_ERROR_INVALID_PTX => "CUDA_ERROR_INVALID_PTX",
        CUresult::CUDA_ERROR_UNKNOWN => "CUDA_ERROR_UNKNOWN",
        CUresult::CUDA_ERROR_NOT_SUPPORTED => "CUDA_ERROR_NOT_SUPPORTED",
        CUresult::CUDA_ERROR_ILLEGAL_STATE => "CUDA_ERROR_ILLEGAL_STATE",
        CUresult::CUDA_ERROR_INVALID_HANDLE => "CUDA_ERROR_INVALID_HANDLE",
        CUresult::CUDA_ERROR_FILE_NOT_FOUND => "CUDA_ERROR_FILE_NOT_FOUND",
        CUresult::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
            "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"
        }
        CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        CUresult::CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED",
        _ => "CUDA_ERROR_UNKNOWN",
    };
    // The names are static, NUL terminated strings.
    let text: &'static [u8] = match text {
        "CUDA_SUCCESS" => b"CUDA_SUCCESS\0",
        "CUDA_ERROR_INVALID_VALUE" => b"CUDA_ERROR_INVALID_VALUE\0",
        "CUDA_ERROR_OUT_OF_MEMORY" => b"CUDA_ERROR_OUT_OF_MEMORY\0",
        "CUDA_ERROR_NOT_INITIALIZED" => b"CUDA_ERROR_NOT_INITIALIZED\0",
        "CUDA_ERROR_INVALID_DEVICE" => b"CUDA_ERROR_INVALID_DEVICE\0",
        "CUDA_ERROR_INVALID_IMAGE" => b"CUDA_ERROR_INVALID_IMAGE\0",
        "CUDA_ERROR_INVALID_CONTEXT" => b"CUDA_ERROR_INVALID_CONTEXT\0",
        "CUDA_ERROR_NOT_FOUND" => b"CUDA_ERROR_NOT_FOUND\0",
        "CUDA_ERROR_NOT_READY" => b"CUDA_ERROR_NOT_READY\0",
        "CUDA_ERROR_LAUNCH_FAILED" => b"CUDA_ERROR_LAUNCH_FAILED\0",
        "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" => b"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES\0",
        "CUDA_ERROR_INVALID_PTX" => b"CUDA_ERROR_INVALID_PTX\0",
        "CUDA_ERROR_NOT_SUPPORTED" => b"CUDA_ERROR_NOT_SUPPORTED\0",
        "CUDA_ERROR_ILLEGAL_STATE" => b"CUDA_ERROR_ILLEGAL_STATE\0",
        "CUDA_ERROR_INVALID_HANDLE" => b"CUDA_ERROR_INVALID_HANDLE\0",
        "CUDA_ERROR_FILE_NOT_FOUND" => b"CUDA_ERROR_FILE_NOT_FOUND\0",
        "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND" => {
            b"CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND\0"
        }
        "CUDA_ERROR_NO_BINARY_FOR_GPU" => b"CUDA_ERROR_NO_BINARY_FOR_GPU\0",
        "CUDA_ERROR_DEINITIALIZED" => b"CUDA_ERROR_DEINITIALIZED\0",
        _ => b"CUDA_ERROR_UNKNOWN\0",
    };
    unsafe { *str = text.as_ptr() as *const i8 };
    CUresult::CUDA_SUCCESS
}

pub(crate) fn get_error_string(error: CUresult, str: *mut *const i8) -> CUresult {
    if str == ptr::null_mut() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let text = stringify_curesult!(
        error => [
            CUDA_SUCCESS,
            CUDA_ERROR_INVALID_VALUE,
            CUDA_ERROR_OUT_OF_MEMORY,
            CUDA_ERROR_NOT_INITIALIZED,
            CUDA_ERROR_DEINITIALIZED,
            CUDA_ERROR_PROFILER_DISABLED,
            CUDA_ERROR_PROFILER_NOT_INITIALIZED,
            CUDA_ERROR_PROFILER_ALREADY_STARTED,
            CUDA_ERROR_PROFILER_ALREADY_STOPPED,
            CUDA_ERROR_NO_DEVICE,
            CUDA_ERROR_INVALID_DEVICE,
            CUDA_ERROR_INVALID_IMAGE,
            CUDA_ERROR_INVALID_CONTEXT,
            CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
            CUDA_ERROR_MAP_FAILED,
            CUDA_ERROR_UNMAP_FAILED,
            CUDA_ERROR_ARRAY_IS_MAPPED,
            CUDA_ERROR_ALREADY_MAPPED,
            CUDA_ERROR_NO_BINARY_FOR_GPU,
            CUDA_ERROR_ALREADY_ACQUIRED,
            CUDA_ERROR_NOT_MAPPED,
            CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
            CUDA_ERROR_NOT_MAPPED_AS_POINTER,
            CUDA_ERROR_ECC_UNCORRECTABLE,
            CUDA_ERROR_UNSUPPORTED_LIMIT,
            CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
            CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
            CUDA_ERROR_INVALID_PTX,
            CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
            CUDA_ERROR_NVLINK_UNCORRECTABLE,
            CUDA_ERROR_JIT_COMPILER_NOT_FOUND,
            CUDA_ERROR_INVALID_SOURCE,
            CUDA_ERROR_FILE_NOT_FOUND,
            CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
            CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
            CUDA_ERROR_OPERATING_SYSTEM,
            CUDA_ERROR_INVALID_HANDLE,
            CUDA_ERROR_ILLEGAL_STATE,
            CUDA_ERROR_NOT_FOUND,
            CUDA_ERROR_NOT_READY,
            CUDA_ERROR_ILLEGAL_ADDRESS,
            CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
            CUDA_ERROR_LAUNCH_TIMEOUT,
            CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
            CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
            CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
            CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
            CUDA_ERROR_CONTEXT_IS_DESTROYED,
            CUDA_ERROR_ASSERT,
            CUDA_ERROR_TOO_MANY_PEERS,
            CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
            CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
            CUDA_ERROR_HARDWARE_STACK_ERROR,
            CUDA_ERROR_ILLEGAL_INSTRUCTION,
            CUDA_ERROR_MISALIGNED_ADDRESS,
            CUDA_ERROR_INVALID_ADDRESS_SPACE,
            CUDA_ERROR_INVALID_PC,
            CUDA_ERROR_LAUNCH_FAILED,
            CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
            CUDA_ERROR_NOT_PERMITTED,
            CUDA_ERROR_NOT_SUPPORTED,
            CUDA_ERROR_SYSTEM_NOT_READY,
            CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
            CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
            CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
            CUDA_ERROR_STREAM_CAPTURE_MERGE,
            CUDA_ERROR_STREAM_CAPTURE_UNMATCHED,
            CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
            CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
            CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
            CUDA_ERROR_CAPTURED_EVENT,
            CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD,
            CUDA_ERROR_TIMEOUT,
            CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE,
            CUDA_ERROR_UNKNOWN
        ]
    );
    match text {
        Some(text) => {
            unsafe { *str = text.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        None => CUresult::CUDA_ERROR_INVALID_VALUE,
    }
}

unsafe fn transmute_lifetime<'a, 'b, T: ?Sized>(t: &'a T) -> &'b T {
    mem::transmute(t)
}

unsafe fn transmute_lifetime_mut<'a, 'b, T: ?Sized>(t: &'a mut T) -> &'b mut T {
    mem::transmute(t)
}

pub fn driver_get_version() -> c_int {
    i32::max_value()
}

impl<'a> CudaRepr for CUctx_st {
    type Impl = context::Context;
}

impl<'a> CudaRepr for CUdevice {
    type Impl = device::Index;
}

impl Decuda<device::Index> for CUdevice {
    fn decuda(self) -> device::Index {
        device::Index(self.0)
    }
}

impl<'a> CudaRepr for CUdeviceptr {
    type Impl = *mut c_void;
}

impl Decuda<*mut c_void> for CUdeviceptr {
    fn decuda(self) -> *mut c_void {
        self.0 as *mut _
    }
}

impl<'a> CudaRepr for CUmod_st {
    type Impl = module::Module;
}

impl<'a> CudaRepr for CUlib_st {
    type Impl = library::Library;
}

impl<'a> CudaRepr for CUkern_st {
    type Impl = library::Kernel;
}

impl<'a> CudaRepr for CUfunc_st {
    type Impl = function::Function;
}

impl<'a> CudaRepr for CUstream_st {
    type Impl = stream::Stream;
}

impl<'a> CudaRepr for CUevent_st {
    type Impl = event::Event;
}

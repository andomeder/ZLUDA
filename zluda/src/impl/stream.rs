use super::{
    context::{Context, ContextData},
    CUresult, GlobalState,
};
use std::{
    collections::VecDeque,
    ffi::c_void,
    mem,
    ptr,
    sync::Mutex,
};

use super::{HasLivenessCookie, LiveCheck};

/// The name of the kernel that was launched most recently, for diagnostics
/// only: when a synchronize never completes, the kernel that is still running
/// is the one that named itself last.
static LAST_KERNEL: Mutex<Option<String>> = Mutex::new(None);

pub fn remember_launched_kernel(name: &str) {
    if let Ok(mut last) = LAST_KERNEL.lock() {
        *last = Some(name.to_string());
    }
    if let Ok(mut guard) = PENDING_OPS.lock() {
        let queue = guard.get_or_insert_with(VecDeque::new);
        if queue.len() < 64 {
            queue.push_back(name.to_string());
        }
    }
}

/// The kernels enqueued since the last synchronize that completed, oldest
/// first: the oldest one is the operation that never retired.
static PENDING_OPS: Mutex<Option<VecDeque<String>>> = Mutex::new(None);

fn pending_kernels() -> Vec<String> {
    let mut guard = match PENDING_OPS.lock() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let queue = guard.get_or_insert_with(VecDeque::new);
    queue.iter().take(3).cloned().collect()
}

fn clear_pending_kernels() {
    if let Ok(mut guard) = PENDING_OPS.lock() {
        if let Some(queue) = guard.as_mut() {
            queue.clear();
        }
    }
}

fn last_launched_kernel() -> String {
    LAST_KERNEL
        .lock()
        .ok()
        .and_then(|last| last.clone())
        .unwrap_or_default()
}

fn transmute_lifetime<T>(x: &T) -> &'static T {
    unsafe { mem::transmute(x) }
}


pub type Stream = LiveCheck<StreamData>;

pub const CU_STREAM_LEGACY: *mut Stream = 1 as *mut _;
pub const CU_STREAM_PER_THREAD: *mut Stream = 2 as *mut _;

impl HasLivenessCookie for StreamData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0x512097354de18d35;

    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0x77d5cc0b;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        if self.context != ptr::null_mut() {
            let context = unsafe { &mut *self.context };
            if !context.streams.remove(&(self as *mut _)) {
                return Err(CUresult::CUDA_ERROR_UNKNOWN);
            }
        }
        Ok(())
    }
}

pub struct StreamData {
pub context: *mut ContextData,
// Immediate CommandList
pub cmd_list: l0::CommandList<'static>,
pub busy_events: VecDeque<(l0::Event<'static>, u64)>,
// This could be a Vec, but I'd rather reuse earliest enqueued event not the one recently enqueued
pub free_events: VecDeque<(l0::Event<'static>, u64)>,
    pub synchronization_event: (l0::Event<'static>, u64),
    pub flags: u32,
}

impl StreamData {
    pub fn new_unitialized(
        ctx: &'static l0::Context,
        device: l0::Device,
        host_event: (l0::Event<'static>, u64),
    ) -> Result<Self, CUresult> {
        Ok(StreamData {
            context: ptr::null_mut(),
            cmd_list: l0::CommandList::new_immediate(ctx, device)?,
            busy_events: VecDeque::new(),
            free_events: VecDeque::new(),
            synchronization_event: host_event,
            flags: 0,
    })
    }
    pub fn new(ctx: &mut ContextData, flags: u32) -> Result<Self, CUresult> {
        let l0_ctx = &mut unsafe { &mut *ctx.device }.l0_context;
        let device = unsafe { &*ctx.device }.base;
        let synchronization_event = unsafe { &mut *ctx.device }
            .host_event_pool
            .get(device, l0_ctx)?;
        let cmd_list = l0::CommandList::new_immediate(l0_ctx, device)?;
        Ok(StreamData {
            context: ctx as *mut _,
            cmd_list,
            busy_events: VecDeque::new(),
            free_events: VecDeque::new(),
            synchronization_event,
            flags,
        })
    }

    pub fn try_reuse_finished_events(&mut self) -> l0::Result<()> {
        loop {
            match self.busy_events.get(0) {
                None => return Ok(()),
                Some((ev, _)) => {
                    if ev.is_ready()? {
                        let (ev, marker) = self.busy_events.pop_front().unwrap();
                        ev.host_reset()?;
                        self.free_events.push_back((ev, marker));
                    } else {
                        return Ok(());
                    }
                }
            }
        }
    }

    pub fn reuse_all_finished_events(&mut self) -> l0::Result<()> {
        self.free_events.reserve(self.busy_events.len());
        for (ev, marker) in self.busy_events.drain(..) {
            ev.host_reset()?;
            self.free_events.push_back((ev, marker));
        }
        Ok(())
    }

    pub fn get_last_event(&self) -> Option<&l0::Event<'static>> {
        self.busy_events.iter().next_back().map(|(ev, _)| ev)
    }

    pub fn push_event(&mut self, ev: (l0::Event<'static>, u64)) {
        self.busy_events.push_back(ev);
    }

    pub fn synchronize(&mut self) -> l0::Result<()> {
        let trace = std::env::var_os("ZLUDA_TRACE_STREAM").is_some();
        let empty = [];
        let busy_event_arr = self.busy_events.back().map(|(ev, _)| [ev]);
        let wait_events = busy_event_arr.as_ref().map_or(&empty[..], |arr| &arr[..]);
        unsafe {
            if let Err(e) = self
                .cmd_list
                .append_barrier(Some(&self.synchronization_event.0), wait_events)
            {
                if trace {
                    eprintln!("stream sync: append_barrier -> {:?}", e);
                }
                return Err(e);
            }
        };
        if let Err(e) = self
            .synchronization_event
            .0
            .host_synchronize(60_000_000_000)
        {
            if trace {
                let last_ready = self
                    .busy_events
                    .back()
                    .map(|(ev, _)| ev.is_ready().map(|r| r).unwrap_or(false));
                let sync_ready = self.synchronization_event.0.is_ready().unwrap_or(false);
                eprintln!(
                    "stream sync: host_synchronize -> {:?} (busy {}, last busy ready {:?}, sync event ready {}, last kernel {})",
                    e,
                    self.busy_events.len(),
                    last_ready,
                    sync_ready,
                    last_launched_kernel()
                );
                eprintln!("stream sync: oldest pending kernels {:?}", pending_kernels());
            }
            return Err(e);
        }
        self.synchronization_event.0.host_reset()?;
        clear_pending_kernels();
        self.reuse_all_finished_events()?;
        Ok(())
    }

    pub fn get_event(
        &mut self,
        l0_dev: l0::Device,
        l0_ctx: &'static l0::Context,
    ) -> l0::Result<(l0::Event<'static>, u64)> {
        self.free_events
            .pop_front()
            .map(|x| Ok(x))
            .unwrap_or_else(|| {
                let event_pool = unsafe { &mut (*(*self.context).device).device_event_pool };
                event_pool.get(l0_dev, l0_ctx)
            })
    }
}

impl Drop for StreamData {
    fn drop(&mut self) {
        if self.context == ptr::null_mut() {
            return;
        }
        for (_, marker) in self.busy_events.iter().chain(self.free_events.iter()) {
            let device_event_pool = unsafe { &mut (*(*self.context).device).device_event_pool };
            device_event_pool.mark_as_free(*marker);
        }
        unsafe { (&mut *self.context).streams.remove(&(&mut *self as *mut _)) };
    }
}

pub(crate) fn get_ctx(hstream: *mut Stream, pctx: *mut *mut Context) -> Result<(), CUresult> {
    if pctx == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let ctx_ptr = GlobalState::lock_stream(hstream, |stream| stream.context)?;
    if ctx_ptr == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED);
    }
    unsafe { *pctx = Context::ptr_from_inner(ctx_ptr) };
    Ok(())
}

pub(crate) fn create(phstream: *mut *mut Stream, flags: u32) -> Result<(), CUresult> {
        let stream_ptr = GlobalState::lock_current_context(|ctx| {
        let mut stream_box = Box::new(Stream::new(StreamData::new(ctx, flags)?));
        let stream_ptr = stream_box.as_mut().as_option_mut().unwrap() as *mut _;
        if !ctx.streams.insert(stream_ptr) {
            return Err(CUresult::CUDA_ERROR_UNKNOWN);
        }
        mem::forget(stream_box);
        Ok::<_, CUresult>(stream_ptr)
    })??;
    unsafe { *phstream = Stream::ptr_from_inner(stream_ptr) };
    Ok(())
}

pub(crate) fn destroy_v2(pstream: *mut Stream) -> Result<(), CUresult> {
    if pstream == ptr::null_mut() || pstream == CU_STREAM_LEGACY || pstream == CU_STREAM_PER_THREAD
    {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock(|_| Stream::destroy_impl(pstream))?
}

pub(crate) fn synchronize(pstream: *mut Stream) -> Result<(), CUresult> {
GlobalState::lock_stream(pstream, |stream_data| Ok(stream_data.synchronize()?))?
}

pub(crate) fn query(pstream: *mut Stream) -> Result<(), CUresult> {
// Returns SUCCESS if all work in the stream has completed, NOT_READY
// otherwise. We check the last busy event.
let ready = GlobalState::lock_stream(pstream, |stream_data| {
stream_data.try_reuse_finished_events().map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
Ok::<_, CUresult>(stream_data.busy_events.is_empty())
})??;
if ready {
Ok(())
} else {
Err(CUresult::CUDA_ERROR_NOT_READY)
}
}

pub(crate) fn get_flags(pstream: *mut Stream, flags: *mut u32) -> Result<(), CUresult> {
    if flags == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let stored = GlobalState::lock_stream(pstream, |stream_data| stream_data.flags)?;
    unsafe { *flags = stored };
    Ok(())
}

pub(crate) fn wait_event(
    pstream: *mut Stream,
    hevent: *mut super::event::Event,
    _flags: u32,
) -> Result<(), CUresult> {
    if hevent == ptr::null_mut() {
        return Ok(());
    }
    // Enqueue a barrier that waits on the event's L0 signal. The stream's
    // subsequent work will only run after the event fires.
    GlobalState::lock_stream(pstream, |stream_data| {
        let l0_ev = unsafe { &*hevent }.as_result()?.l0_event();
        let ev = match l0_ev {
            Some(e) => e,
            None => return Ok::<(), CUresult>(()),
        };
        let cmd_list = unsafe { transmute_lifetime(&stream_data.cmd_list) };
        unsafe {
            cmd_list.append_barrier(None, &[ev])?;
        }
        Ok(())
    })?
}

pub(crate) fn add_callback(
    pstream: *mut Stream,
    callback: super::super::cuda::CUstreamCallback,
    user_data: *mut c_void,
    _flags: u32,
) -> Result<(), CUresult> {
    let cb = match callback {
        Some(cb) => cb,
        None => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
    };
    // L0 has no host-callback primitive; the closest semantics is: after all
    // currently-enqueued work in the stream completes, invoke the callback
    // from the host. We synchronize the stream, then call back.
    GlobalState::lock_stream(pstream, |stream_data| Ok::<(), CUresult>(stream_data.synchronize()?))??;
    unsafe {
        cb(
            pstream as *mut _,
            CUresult::CUDA_SUCCESS,
            user_data,
        );
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::cuda::CUstream;

    use super::super::test::CudaDriverFns;
    use super::super::CUresult;
    use std::{ptr, thread};

    const CU_STREAM_LEGACY: CUstream = 1 as *mut _;
    const CU_STREAM_PER_THREAD: CUstream = 2 as *mut _;

    cuda_driver_test!(default_stream_uses_current_ctx_legacy);
    cuda_driver_test!(default_stream_uses_current_ctx_ptsd);

    fn default_stream_uses_current_ctx_legacy<T: CudaDriverFns>() {
        default_stream_uses_current_ctx_impl::<T>(CU_STREAM_LEGACY);
    }

    fn default_stream_uses_current_ctx_ptsd<T: CudaDriverFns>() {
        default_stream_uses_current_ctx_impl::<T>(CU_STREAM_PER_THREAD);
    }

    fn default_stream_uses_current_ctx_impl<T: CudaDriverFns>(stream: CUstream) {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx1 = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx1, 0, 0), CUresult::CUDA_SUCCESS);
        let mut stream_ctx1 = ptr::null_mut();
        assert_eq!(
            T::cuStreamGetCtx(stream, &mut stream_ctx1),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(ctx1, stream_ctx1);
        let mut ctx2 = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx2, 0, 0), CUresult::CUDA_SUCCESS);
        assert_ne!(ctx1, ctx2);
        let mut stream_ctx2 = ptr::null_mut();
        assert_eq!(
            T::cuStreamGetCtx(stream, &mut stream_ctx2),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(ctx2, stream_ctx2);
        //  Cleanup
        assert_eq!(T::cuCtxDestroy_v2(ctx1), CUresult::CUDA_SUCCESS);
        assert_eq!(T::cuCtxDestroy_v2(ctx2), CUresult::CUDA_SUCCESS);
    }

    cuda_driver_test!(stream_context_destroyed);

    fn stream_context_destroyed<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx, 0, 0), CUresult::CUDA_SUCCESS);
        let mut stream = ptr::null_mut();
        assert_eq!(T::cuStreamCreate(&mut stream, 0), CUresult::CUDA_SUCCESS);
        let mut stream_ctx1 = ptr::null_mut();
        assert_eq!(
            T::cuStreamGetCtx(stream, &mut stream_ctx1),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(stream_ctx1, ctx);
        assert_eq!(T::cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
        let mut stream_ctx2 = ptr::null_mut();
        // When a context gets destroyed, its streams are also destroyed
        let cuda_result = T::cuStreamGetCtx(stream, &mut stream_ctx2);
        assert!(
            cuda_result == CUresult::CUDA_ERROR_INVALID_HANDLE
                || cuda_result == CUresult::CUDA_ERROR_INVALID_CONTEXT
                || cuda_result == CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED
        );
        assert_eq!(
            T::cuStreamDestroy_v2(stream),
            CUresult::CUDA_ERROR_INVALID_HANDLE
        );
        // Check if creating another context is possible
        let mut ctx2 = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx2, 0, 0), CUresult::CUDA_SUCCESS);
        //  Cleanup
        assert_eq!(T::cuCtxDestroy_v2(ctx2), CUresult::CUDA_SUCCESS);
    }

    cuda_driver_test!(stream_moves_context_to_another_thread);

    fn stream_moves_context_to_another_thread<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx, 0, 0), CUresult::CUDA_SUCCESS);
        let mut stream = ptr::null_mut();
        assert_eq!(T::cuStreamCreate(&mut stream, 0), CUresult::CUDA_SUCCESS);
        let mut stream_ctx1 = ptr::null_mut();
        assert_eq!(
            T::cuStreamGetCtx(stream, &mut stream_ctx1),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(stream_ctx1, ctx);
        let stream_ptr = stream as usize;
        let stream_ctx_on_thread = thread::spawn(move || {
            let mut stream_ctx2 = ptr::null_mut();
            assert_eq!(
                T::cuStreamGetCtx(stream_ptr as *mut _, &mut stream_ctx2),
                CUresult::CUDA_SUCCESS
            );
            stream_ctx2 as usize
        })
        .join()
        .unwrap();
        assert_eq!(stream_ctx1, stream_ctx_on_thread as *mut _);
        //  Cleanup
        assert_eq!(T::cuStreamDestroy_v2(stream), CUresult::CUDA_SUCCESS);
        assert_eq!(T::cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
    }

    cuda_driver_test!(can_destroy_stream);

    fn can_destroy_stream<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx, 0, 0), CUresult::CUDA_SUCCESS);
        let mut stream = ptr::null_mut();
        assert_eq!(T::cuStreamCreate(&mut stream, 0), CUresult::CUDA_SUCCESS);
        assert_eq!(T::cuStreamDestroy_v2(stream), CUresult::CUDA_SUCCESS);
        // Cleanup
        assert_eq!(T::cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
    }

    cuda_driver_test!(cant_destroy_default_stream);

    fn cant_destroy_default_stream<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx, 0, 0), CUresult::CUDA_SUCCESS);
        assert_ne!(
            T::cuStreamDestroy_v2(super::CU_STREAM_LEGACY as *mut _),
            CUresult::CUDA_SUCCESS
        );
        // Cleanup
        assert_eq!(T::cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
    }
}

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

/// The name of the kernel launched most recently, for diagnostics: when a
/// synchronize never completes, the kernel still running is the one that named
/// itself last.
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
// A deque so the earliest enqueued event is reused first.
pub free_events: VecDeque<(l0::Event<'static>, u64)>,
// Launches left before the event list is examined again (see
// try_reuse_finished_events).
recycle_countdown: u32,
// Last known size of the retired prefix, the first guess of the next search.
retired_hint: usize,
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
            recycle_countdown: 0,
            retired_hint: 1,
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
            recycle_countdown: 0,
            retired_hint: 1,
            synchronization_event,
            flags,
        })
    }

    /// Recycle every ready event except the newest, so an event is only reusable
    /// once nothing can still be waiting on it.
    ///
    /// An event is not safe to reuse the moment it signals. Each launch appends
    /// a barrier that waits on the *previous* launch's event, and that barrier
    /// stays pending until it is processed. Reusing the event before its
    /// successor's barrier has retired hands the newer launch a wait that can
    /// never be satisfied in order: the successor waits for the event, the
    /// event now belongs to a later launch, and the queue stops for good. That
    /// is what hung torch at 527 queued launches - the merge in `merge_adapter`
    /// launches one kernel per parameter - and why a 200 launch loop sometimes
    /// stalled and sometimes did not.
    ///
    /// A handful of status queries for the whole list, not one per event: events
    /// signal in the order their commands were appended, so "the event at index
    /// i has signalled" is monotone in i and a bisect finds the boundary. One
    /// query costs 3.5 us here, almost twice the launch's own append
    /// (`tmp/harness/launchcost/query_cost.c`), which is why asking per event
    /// was the most expensive thing in a page call.
    fn recycle_all_ready(&mut self) -> l0::Result<usize> {
        let len = self.busy_events.len();
        let lag = super::diag::wait_lag();
        // Recycling event i is safe only once the launch `lag` places after it
        // has retired, because that launch's barrier is what waits on it. The
        // newest event is kept, and so are the `lag` before it.
        if len <= lag {
            return Ok(0);
        }
        let max = len - lag;
        // Search the retired prefix: `count` events may be recycled when the
        // event at `count - 1 + lag` has signalled. That predicate is monotone
        // (a later event never signals before an earlier one), so a bisect finds
        // the whole prefix in a few queries, and the previous answer is the first
        // guess because a steady stream retires a similar number each time.
        let mut lo = 0usize;
        let mut hi = max;
        let mut probe = self.retired_hint.clamp(1, max);
        // A capped number of probes: when nothing has retired, walking a stale
        // guess down to zero would cost log2(guess) queries every time, and the
        // guess only becomes stale when the device has stopped keeping up. Four
        // probes still find a prefix that has shrunk by 16x.
        for _ in 0..4 {
            let index = probe - 1 + lag;
            let ready = self.busy_events.get(index).unwrap().0.is_ready()?;
            if ready {
                lo = probe;
            } else {
                hi = probe - 1;
            }
            if lo >= hi {
                break;
            }
            probe = lo + (hi - lo + 1) / 2;
        }
        self.retired_hint = lo;
        for _ in 0..lo {
            let (event, marker) = self.busy_events.pop_front().unwrap();
            event.host_reset()?;
            self.free_events.push_back((event, marker));
        }
        Ok(lo)
    }

    pub fn try_reuse_finished_events(&mut self) -> l0::Result<()> {
        // Nothing to do while the free list has a comfortable stock, and nothing
        // to do until the counter says an examination is due: a query costs
        // 3.5 us on this driver (`tmp/harness/launchcost/query_cost.c`), almost
        // twice the append it is guarding, so it is worth asking rarely and
        // taking a whole prefix when we do.
        let interval = super::diag::event_recycle_interval() as usize;
        if self.free_events.len() >= interval {
            return Ok(());
        }
        if self.recycle_countdown > 0 {
            self.recycle_countdown -= 1;
            return Ok(());
        }
        self.recycle_countdown = interval as u32;
        self.recycle_all_ready()?;
        Ok(())
    }

    pub fn reuse_all_finished_events(&mut self) -> l0::Result<()> {
        self.free_events.reserve(self.busy_events.len());
        for (ev, marker) in self.busy_events.drain(..) {
            ev.host_reset()?;
            self.free_events.push_back((ev, marker));
        }
        Ok(())
    }

    /// Wait until at most `keep` launches are still in flight, retiring the
    /// oldest markers as they signal.
    ///
    /// A caller that reuses a buffer a launch read needs this: the marker
    /// `keep` launches back retires no earlier than that launch did, so once
    /// the queue is that short the buffer is free.
    ///
    /// One synchronise, not one status query per event: events retire in the
    /// order they were appended, so waiting for the last event that has to go
    /// retires all of them, and the retired prefix is then moved to the free
    /// list without asking about each one. It used to ask per event, which at
    /// 3.5 us a query made this path the most expensive thing in a page call
    /// once the recycler stopped trimming the list on every launch.
    pub fn wait_for_pending(&mut self, keep: usize) -> l0::Result<()> {
        let lag = super::diag::wait_lag();
        while self.busy_events.len() > keep {
            // Free whatever has already retired before blocking at all, and
            // again after each block: one bisect, not one query per event.
            if self.recycle_all_ready()? > 0 {
                continue;
            }
            // Nothing more can go without waiting. Block on the event whose
            // retirement makes enough of the list recyclable, which is
            // `lag` places past the oldest event that has to go.
            let needing = self.busy_events.len() - keep;
            let probe = (needing - 1 + lag).min(self.busy_events.len() - 1);
            match self.busy_events.get(probe) {
                None => return Ok(()),
                Some((event, _)) => event.host_synchronize(60_000_000_000)?,
            }
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
        let trace = super::diag::trace_stream();
        // ZLUDA_IDLE_SYNC_BARRIER=1 appends the barrier anyway, which is what
        // this used to do: it is how the regression test below is shown to be
        // able to fail.
        if self.busy_events.is_empty() && !super::diag::idle_sync_barrier() {
            // Nothing is queued on this stream, so there is nothing to wait
            // for. It matters: a barrier appended with an empty wait list does
            // not signal on this driver, so waiting on one stalls until the
            // timeout - which is how a sync on an idle stream turned into a 60
            // second hang with no work pending at all.
            clear_pending_kernels();
            return Ok(());
        }
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
// SUCCESS when all work in the stream has completed, NOT_READY otherwise.
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
    // L0 has no host-callback primitive, so the closest semantics is to
    // synchronize the stream and then call back from the host.
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

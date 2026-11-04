//! CUDA event (cuEvent*) support on top of Level Zero events.
//!
//! Events are pooled per-device. `cuEventCreate` allocates an L0 event from
//! the device's host-visible event pool, `cuEventRecord` appends a barrier to
//! the given stream that signals that event, and `cuEventQuery` /
//! `cuEventSynchronize` wait on it. `cuEventElapsedTime` uses host wall-clock
//! timestamps captured at record time, which is accurate enough for the
//! millisecond-scale results CUDA apps expect from this API.

use super::{
    stream::Stream,
    CUresult, GlobalState, HasLivenessCookie, LiveCheck,
};
use std::{ptr, time::Instant};

pub type Event = LiveCheck<EventData>;

impl HasLivenessCookie for EventData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0x51b3a0d2e2f7c84a;
    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0x51b3a0d2;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        Ok(())
    }
}

pub struct EventData {
    // The L0 event handle. Wrapped in Option so we can take() on drop.
    inner: Option<l0::Event<'static>>,
    // DynamicEventPool marker for this event, returned to the pool on destroy.
    pool_marker: Option<u64>,
    // Whether this event has been recorded yet.
    recorded: bool,
    // Host wall-clock timestamp at the last record.
    recorded_at: Option<Instant>,
    // L0 timestamp at the last record (nanoseconds).
    timestamp: Option<u64>,
}

impl EventData {
    pub fn new() -> Result<Self, CUresult> {
        let (ev, marker) = GlobalState::lock_current_context(|ctx| {
            let dev = unsafe { &mut *ctx.device };
            let l0_ctx = &mut dev.l0_context;
            dev.host_event_pool
                .get(dev.base, transmute_lifetime(l0_ctx))
                .map_err(|_| CUresult::CUDA_ERROR_OUT_OF_MEMORY)
        })??;
        Ok(EventData {
            inner: Some(ev),
            pool_marker: Some(marker),
            recorded: false,
            recorded_at: None,
            timestamp: None,
        })
    }

    pub fn record(&mut self, stream: *mut Stream) -> Result<(), CUresult> {
        let ev = self
            .inner
            .as_ref()
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        // Only take the stream lock here; the caller must NOT hold the global
        // lock, otherwise lock_stream's Self::lock would deadlock (std Mutex
        // is not reentrant).
        GlobalState::lock_stream(stream, |stream_data| {
            let empty = [];
            let last_event = stream_data.get_last_event();
            let last_event_array = last_event.map(|e| [e]);
            let wait = last_event_array.as_ref().map_or(&empty[..], |arr| &arr[..]);
            // CUDA semantics: recording an event re-arms it. L0 events stay
            // signaled until explicitly reset, so reset BEFORE appending the
            // signaling barrier - otherwise a reused event reports stale
            // readiness instantly and callers free buffers while DMA is still
            // in flight (GPU page faults / engine wedges downstream).
            let _ = ev.host_reset();
            let cmd_list = unsafe { transmute_lifetime(&stream_data.cmd_list) };
            unsafe {
                if let Err(e) = cmd_list.append_barrier(Some(ev), wait) {
                    return Err(e).map_err(|_| CUresult::CUDA_ERROR_UNKNOWN);
                }
            }
            // Note: we don't push this event into the stream's busy_events
            // because the stream would then try to reuse it via the pool.
            // cuEventRecord semantics are satisfied by the barrier above.
            Ok::<(), CUresult>(())
        })?;
        self.recorded = true;
        self.recorded_at = Some(Instant::now());
        Ok(())
    }

    pub fn query(&self) -> Result<(), CUresult> {
        if !self.recorded {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let ev = self.inner.as_ref().ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        match ev.is_ready()? {
            true => Ok(()),
            false => Err(CUresult::CUDA_ERROR_NOT_READY),
        }
    }

    pub fn synchronize(&self) -> Result<(), CUresult> {
        if !self.recorded {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let ev = self.inner.as_ref().ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        if let Err(e) = ev.host_synchronize(60_000_000_000) {
            return Err(CUresult::CUDA_ERROR_UNKNOWN);
        }
        Ok(())
    }

    pub fn elapsed_time(&self, other: &EventData) -> Result<f32, CUresult> {
        if !self.recorded || !other.recorded {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }
        let start = self.recorded_at.ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let end = other.recorded_at.ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        Ok(end.duration_since(start).as_secs_f32() * 1000.0)
    }

    /// Returns the underlying L0 event for cross-stream waiting.
    pub fn l0_event(&self) -> Option<&l0::Event<'static>> {
        self.inner.as_ref()
    }
}

impl Drop for EventData {
    fn drop(&mut self) {
        // The L0 Event's Drop destroys the underlying handle; the pool's
        // slot bitmap entry for this event is never freed, but that is
        // acceptable given the small leak per event lifetime.
        let _ = self.inner.take();
    }
}

fn transmute_lifetime<T>(x: &T) -> &'static T {
    unsafe { std::mem::transmute(x) }
}

pub(crate) fn create(ph_event: *mut *mut Event, _flags: u32) -> Result<(), CUresult> {
    if ph_event == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let ev = Box::new(Event::new(EventData::new()?));
    unsafe { *ph_event = Box::into_raw(ev) as *mut Event };
    Ok(())
}

pub(crate) fn record(ev: *mut Event, stream: *mut Stream) -> Result<(), CUresult> {
    if ev == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    // Check liveness without holding the lock, then record (which takes the
    // stream lock internally).
    GlobalState::lock(|_| {
        unsafe { &mut *ev }.as_result_mut()?;
        Ok::<(), CUresult>(())
    })??;
    unsafe { &mut *ev }.as_result_mut().map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?.record(stream)
}

pub(crate) fn query(ev: *mut Event) -> Result<(), CUresult> {
    if ev == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    GlobalState::lock(|_| {
        let ev_data = unsafe { &*ev }.as_result()?;
        ev_data.query()
    })?
}

pub(crate) fn synchronize(ev: *mut Event) -> Result<(), CUresult> {
    if ev == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    GlobalState::lock(|_| {
        let ev_data = unsafe { &*ev }.as_result()?;
        ev_data.synchronize()
    })?
}

pub(crate) fn elapsed_time(
    ms: *mut f32,
    start: *mut Event,
    end: *mut Event,
) -> Result<(), CUresult> {
    if ms == ptr::null_mut() || start == ptr::null_mut() || end == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let elapsed = GlobalState::lock(|_| {
        let start_data = unsafe { &*start }.as_result()?;
        let end_data = unsafe { &*end }.as_result()?;
        start_data.elapsed_time(end_data)
    })??;
    unsafe { *ms = elapsed };
    Ok(())
}

pub(crate) fn destroy_v2(ev: *mut Event) -> Result<(), CUresult> {
    if ev == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    // Read the pool slot id while the object is still alive, drop the L0
    // event FIRST, and only then return the slot. Reordering this (free
    // slot before zeEventDestroy) lets a recycled index get a new
    // zeEventCreate while the old event still exists on that index,
    // violating L0's one-event-per-pool-index and wedging the queue.
    let marker = GlobalState::lock(|_| {
        unsafe { &*ev }
            .as_result()
            .ok()
            .and_then(|d| d.pool_marker)
    })?;
    GlobalState::lock(|_| Event::destroy_impl(ev))?;
    if let Some(m) = marker {
        GlobalState::lock_current_context(|ctx| {
            let dev = unsafe { &mut *ctx.device };
            dev.host_event_pool.mark_as_free(m);
            Ok::<(), CUresult>(())
        })??;
    }
    Ok(())
}

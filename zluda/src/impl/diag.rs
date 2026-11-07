// The diagnostic switches of the launch path, read once.
//
// Every one of these used to be a `std::env::var_os` call on the launch path, and
// a launch is the unit this driver is priced in: a page call is about 44000
// launches, and `std::env::var_os` walks the whole environment block each time.
// The values cannot change inside a process that has already started, so they are
// read on the first use and kept. A `OnceLock` load is a few nanoseconds where the
// lookup is tens, and it keeps a process with a large environment (torch sets
// dozens of variables) from paying it per launch.
use std::sync::OnceLock;

fn flag(set: &'static OnceLock<bool>, name: &'static str) -> bool {
    *set.get_or_init(|| std::env::var_os(name).is_some())
}

pub static TRACE_FUNC: OnceLock<bool> = OnceLock::new();
pub static TRACE_STAGE: OnceLock<bool> = OnceLock::new();
pub static TRACE_STREAM: OnceLock<bool> = OnceLock::new();
pub static TRACE_LAUNCH_MS: OnceLock<bool> = OnceLock::new();
pub static TRACE_SHARED: OnceLock<bool> = OnceLock::new();
pub static PARAM_BLOCKING: OnceLock<bool> = OnceLock::new();
pub static PARAM_HOST: OnceLock<bool> = OnceLock::new();
pub static NO_LAUNCH_WAIT: OnceLock<bool> = OnceLock::new();
pub static SYNC_LAUNCH: OnceLock<bool> = OnceLock::new();
pub static IDLE_SYNC_BARRIER: OnceLock<bool> = OnceLock::new();
pub static RECORD_WAIT: OnceLock<bool> = OnceLock::new();
static DUMP_PARAMS: OnceLock<Option<String>> = OnceLock::new();
static WAIT_LAG: OnceLock<usize> = OnceLock::new();
static EVENT_RECYCLE: OnceLock<u32> = OnceLock::new();
static PARAM_SLOTS: OnceLock<usize> = OnceLock::new();

/// ZLUDA_TRACE_FUNC=1 prints every launch and every launch failure.
pub fn trace_func() -> bool {
    flag(&TRACE_FUNC, "ZLUDA_TRACE_FUNC")
}

/// ZLUDA_TRACE_STAGE=1 prints the argument buffer staging steps.
pub fn trace_stage() -> bool {
    flag(&TRACE_STAGE, "ZLUDA_TRACE_STAGE")
}

/// ZLUDA_TRACE_STREAM=1 records the kernels launched since the last completed
/// synchronize, so a stall can name the operation that never retired.
pub fn trace_stream() -> bool {
    flag(&TRACE_STREAM, "ZLUDA_TRACE_STREAM")
}

/// ZLUDA_TRACE_LAUNCH_MS=1 prints a line per launch, =2 prints a summary every
/// thousand (see launch_trace.rs).
pub fn trace_launch_ms() -> bool {
    *TRACE_LAUNCH_MS.get_or_init(|| std::env::var_os("ZLUDA_TRACE_LAUNCH_MS").is_some())
}

/// ZLUDA_PARAM_BLOCKING=1 fills an oversized argument block with a blocking copy
/// that drains the context, instead of staging it on the launch's own stream.
pub fn param_blocking() -> bool {
    flag(&PARAM_BLOCKING, "ZLUDA_PARAM_BLOCKING")
}

/// ZLUDA_PARAM_HOST=1 keeps that block in host memory instead of device memory.
pub fn param_host() -> bool {
    flag(&PARAM_HOST, "ZLUDA_PARAM_HOST")
}

/// ZLUDA_NO_LAUNCH_WAIT=1 drops the wait on the previous launch. Only correct
/// with a command list created in order; see tmp/harness/launchcost.
pub fn no_launch_wait() -> bool {
    flag(&NO_LAUNCH_WAIT, "ZLUDA_NO_LAUNCH_WAIT")
}

/// ZLUDA_SYNC_LAUNCH=1 blocks until each launch retires.
pub fn sync_launch() -> bool {
    flag(&SYNC_LAUNCH, "ZLUDA_SYNC_LAUNCH")
}

/// ZLUDA_IDLE_SYNC_BARRIER=1 appends the synchronize barrier even when the
/// stream has nothing queued. A regression test for the idle-sync hang.
pub fn idle_sync_barrier() -> bool {
    flag(&IDLE_SYNC_BARRIER, "ZLUDA_IDLE_SYNC_BARRIER")
}

/// ZLUDA_RECORD_WAIT=1 makes cuEventRecord wait on the stream's last event.
pub fn record_wait() -> bool {
    flag(&RECORD_WAIT, "ZLUDA_RECORD_WAIT")
}

/// ZLUDA_TRACE_SHARED=1 prints the shared memory a launch asks the engine for,
/// split into the prefix the kernel's own addressing assumes and the
/// application's own dynamic request.
pub fn trace_shared() -> bool {
    flag(&TRACE_SHARED, "ZLUDA_TRACE_SHARED")
}

/// ZLUDA_DUMP_PARAMS=<substring> prints the bytes of every argument of a launch
/// whose kernel name matches.
pub fn dump_params(name: &str) -> bool {
    DUMP_PARAMS
        .get_or_init(|| std::env::var("ZLUDA_DUMP_PARAMS").ok())
        .as_ref()
        .map(|want| name.contains(want.as_str()))
        .unwrap_or(false)
}

/// ZLUDA_WAIT_LAG=N makes a new command wait N commands back (default 1).
pub fn wait_lag() -> usize {
    *WAIT_LAG.get_or_init(|| {
        std::env::var("ZLUDA_WAIT_LAG")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value >= 1)
            .unwrap_or(1)
    })
}

/// ZLUDA_EVENT_RECYCLE=N: launches between two examinations of the stream's
/// event list (default 8, 1 restores a query per launch).
pub fn event_recycle_interval() -> u32 {
    *EVENT_RECYCLE.get_or_init(|| {
        std::env::var("ZLUDA_EVENT_RECYCLE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value >= 1)
            .unwrap_or(8)
    })
}

/// ZLUDA_PARAM_SLOTS=N: how many argument-buffer slots a stream may have in
/// flight (default 8).
pub fn param_slots() -> usize {
    *PARAM_SLOTS.get_or_init(|| {
        std::env::var("ZLUDA_PARAM_SLOTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value >= 2)
            .unwrap_or(super::param_buffer::SLOTS)
    })
}

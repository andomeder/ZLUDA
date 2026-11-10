//! CUDA graph capture and replay.
//!
//! Level Zero has no graph API, so a capture records the operations an
//! application submitted while its stream was capturing, and a launch of the
//! instantiated graph re-appends them to a stream's immediate command list.
//!
//! What this buys is compatibility, not speed. The dispatch floor here is
//! 0.34-0.85 us per launch, and a re-append design pays it again, so a replay is
//! worth about 1.0-1.05x (`odocs/PLAN-GRAPHS.md` measured that before this was
//! built). What was missing is that an application using
//! `torch.cuda.CUDAGraph` or `torch.compile(mode="reduce-overhead")` could not
//! run at all: every capture entry point returned CUDA_ERROR_NOT_SUPPORTED.
//!
//! Two of CUDA's contracts are load bearing and are why this cannot be a thin
//! wrapper over "run the ops again":
//!
//! * **A kernel's arguments are copied by value at capture time.** The whole
//!   point of a graph is that the application reuses the same static buffers and
//!   the same launch sites, and changes what they hold between replays. Reading
//!   the caller's pointer again at replay would pick up a stack slot that has
//!   been reused, which is silent corruption of exactly the kind this project
//!   treats as a defect. The bytes are copied here, once.
//! * **Nothing executes during capture.** The operations are recorded and run
//!   when the graph is launched. An application that captures and then reads a
//!   result sees whatever was in the buffer already, which is what real CUDA
//!   gives it too.
//!
//! The rest of this file is the recording side (`record_*`, called from the
//! launch, copy, fill and event paths), the handle types, and replay.

use super::{event, function, stream, CUresult, HasLivenessCookie, LiveCheck};
use crate::cuda::{CUstreamCaptureMode, CUstreamCaptureStatus};
use std::{
    ffi::c_void,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex,
    },
};

pub type Graph = LiveCheck<GraphData>;
pub type GraphExec = LiveCheck<GraphExecData>;

/// One recorded operation.
///
/// Everything is stored as plain integers rather than pointers so the capture
/// can live in a static: a raw pointer is not `Send`. Each one is checked for
/// liveness when it is replayed, so a module destroyed between capture and
/// replay is an error rather than a wild write.
enum Op {
    Launch(Box<Launch>),
    Copy {
        dst: usize,
        src: usize,
        bytes: usize,
    },
    Fill {
        dst: usize,
        pattern: [u8; 4],
        pattern_size: usize,
        bytes: usize,
    },
    EventRecord {
        event: usize,
    },
    EventWait {
        event: usize,
    },
    /// A capture of nothing still produces a graph, and a graph with no nodes
    /// launches as a no-op. Real CUDA does the same.
    Empty,
}

struct Launch {
    /// `*mut function::Function`, checked on every replay.
    func: usize,
    /// For the trace and for error messages.
    name: String,
    grid: [u32; 3],
    block: [u32; 3],
    /// The bytes of each argument, copied out of the application's storage at
    /// capture time. Empty for a kernel that reads its arguments from a block.
    args: Vec<Vec<u8>>,
    /// The argument block, for a kernel whose argument list is larger than the
    /// compiler accepts: the launch passes one pointer, and the kernel reads
    /// everything from there. The bytes are the same by-value copy as `args`.
    param_bytes: Option<Vec<u8>>,
    /// The device buffer holding `param_bytes`. Allocated at instantiate time, so
    /// it belongs to the executable: destroying the graph does not free it and
    /// destroying the executable does.
    param_device: usize,
    /// The shared-memory window this launch asks for, when the kernel uses one.
    /// It depends only on the block shape and the application's own request, and
    /// both are fixed by the capture.
    shared_window: Option<usize>,
    /// The argument index the shared-memory size goes in.
    shared_arg_index: u32,
}

struct Capture {
    /// The stream being captured, as `stream_key` spells it.
    stream: usize,
    mode: CUstreamCaptureMode,
    id: u64,
    ops: Vec<Op>,
}

static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
/// A cheap first test, so the launch path pays one relaxed load rather than a
/// mutex acquisition for every launch outside a capture.
static ACTIVE: AtomicBool = AtomicBool::new(false);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The key a stream is recorded under. The default, legacy and per-thread
/// streams all mean the context's own stream, so they share one key; a created
/// stream is identified by its handle.
fn stream_key(hstream: *mut stream::Stream) -> usize {
    if hstream.is_null()
        || hstream == stream::CU_STREAM_LEGACY
        || hstream == stream::CU_STREAM_PER_THREAD
    {
        0
    } else {
        hstream as usize
    }
}

/// Is anything being captured?
pub fn capturing_any() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Must this stream's operations be recorded instead of submitted?
///
/// Returns an error for an operation submitted to a stream that is not part of
/// the capture in progress. CUDA's thread-local mode requires exactly that, and
/// its global mode would require the operation to merge into the graph, which
/// this implementation does not do: refusing loudly is better than recording an
/// operation whose ordering against the rest of the graph we cannot express.
pub fn recording(hstream: *mut stream::Stream) -> Result<bool, CUresult> {
    if !capturing_any() {
        return Ok(false);
    }
    let guard = CAPTURE.lock().unwrap_or_else(|poison| poison.into_inner());
    match guard.as_ref() {
        None => Ok(false),
        Some(capture) if capture.stream == stream_key(hstream) => Ok(true),
        Some(_) => Err(CUresult::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT),
    }
}

fn with_capture<T>(f: impl FnOnce(&mut Capture) -> Result<T, CUresult>) -> Result<T, CUresult> {
    let mut guard = CAPTURE.lock().unwrap_or_else(|poison| poison.into_inner());
    let capture = guard
        .as_mut()
        .ok_or(CUresult::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED)?;
    f(capture)
}

/// Record a kernel launch. Returns having stored everything the replay needs.
pub fn record_launch(
    func: *mut function::Function,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem_bytes: u32,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<(), CUresult> {
    let func_data = unsafe { &mut *func }
        .as_result_mut()
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?;
    let name = func_data.name.to_string_lossy().into_owned();
    // A kernel that reads its arguments from a block takes one pointer instead;
    // the block itself is the by-value copy.
    let (param_bytes, args) = match func_data.param_buffer.as_ref() {
        Some(layout) => (
            Some(super::param_buffer::collect(layout, kernel_params, extra)?),
            Vec::new(),
        ),
        None => (
            None,
            function::capture_arguments(func_data, kernel_params, extra)?,
        ),
    };
    let shared_window = function::shared_window_for(func_data, shared_mem_bytes, block);
    let own_parameters = if param_bytes.is_some() {
        1
    } else {
        args.len()
    };
    let launch = Launch {
        func: func as usize,
        name,
        grid,
        block,
        args,
        param_bytes,
        param_device: 0,
        shared_window,
        shared_arg_index: own_parameters as u32,
    };
    with_capture(|capture| {
        capture.ops.push(Op::Launch(Box::new(launch)));
        Ok(())
    })
}

/// Record an asynchronous memory copy. The pointers are the ones the
/// application passed: CUDA reads them again on every replay, which is why the
/// application has to keep the buffers alive and why a copy from pageable host
/// memory is refused rather than recorded.
pub fn record_copy(dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<(), CUresult> {
    with_capture(|capture| {
        capture.ops.push(Op::Copy {
            dst: dst as usize,
            src: src as usize,
            bytes,
        });
        Ok(())
    })
}

/// Record a fill. The pattern lives on the caller's stack, so its bytes are
/// copied here.
pub fn record_fill(
    dst: *mut c_void,
    pattern: *const c_void,
    pattern_size: usize,
    bytes: usize,
) -> Result<(), CUresult> {
    if pattern.is_null() || pattern_size > 4 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut stored = [0u8; 4];
    unsafe {
        ptr::copy_nonoverlapping(pattern as *const u8, stored.as_mut_ptr(), pattern_size);
    }
    with_capture(|capture| {
        capture.ops.push(Op::Fill {
            dst: dst as usize,
            pattern: stored,
            pattern_size,
            bytes,
        });
        Ok(())
    })
}

/// Record `cuEventRecord`. The event is not touched: there is no submission to
/// signal it, and a replay is what arms it.
pub fn record_event_record(hev: *mut event::Event) -> Result<(), CUresult> {
    if let Ok(data) = unsafe { &*hev }.as_result() {
        if let Some(l0_event) = data.l0_event() {
            // Clear a signal left over from an earlier record, so a query
            // between capture and the first replay answers NOT_READY instead of
            // reporting stale readiness.
            let _ = l0_event.host_reset();
        }
    }
    with_capture(|capture| {
        capture.ops.push(Op::EventRecord {
            event: hev as usize,
        });
        Ok(())
    })
}

/// Record `cuStreamWaitEvent`.
pub fn record_event_wait(hev: *mut event::Event) -> Result<(), CUresult> {
    with_capture(|capture| {
        capture.ops.push(Op::EventWait {
            event: hev as usize,
        });
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

pub struct GraphData {
    ops: Vec<Op>,
    closed: bool,
}

impl HasLivenessCookie for GraphData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0x47726170684c6f67;
    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0x67726170;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        self.closed = true;
        Ok(())
    }
}

pub struct GraphExecData {
    ops: Vec<Op>,
    flags: u64,
    closed: bool,
}

impl HasLivenessCookie for GraphExecData {
    #[cfg(target_pointer_width = "64")]
    const COOKIE: usize = 0x4772617068457865;
    #[cfg(target_pointer_width = "32")]
    const COOKIE: usize = 0x47726170;

    const LIVENESS_FAIL: CUresult = CUresult::CUDA_ERROR_INVALID_HANDLE;

    fn try_drop(&mut self) -> Result<(), CUresult> {
        self.closed = true;
        // The argument blocks belong to the executable, so this is where they go
        // back. A failure here means the context is already gone, which is the
        // normal case at process teardown; there is nothing to report then.
        for op in self.ops.iter() {
            if let Op::Launch(launch) = op {
                if launch.param_device != 0 {
                    let _ = super::memory::free_v2(launch.param_device as *mut c_void);
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Capture control
// ---------------------------------------------------------------------------

pub fn begin_capture(hstream: *mut stream::Stream, mode: CUstreamCaptureMode) -> Result<(), CUresult> {
    let mut guard = CAPTURE.lock().unwrap_or_else(|poison| poison.into_inner());
    if guard.is_some() {
        // CUDA allows one capture at a time in a process. Two would need the
        // graph to be identified by more than the stream.
        return Err(CUresult::CUDA_ERROR_ILLEGAL_STATE);
    }
    *guard = Some(Capture {
        stream: stream_key(hstream),
        mode,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        ops: Vec::new(),
    });
    ACTIVE.store(true, Ordering::Relaxed);
    Ok(())
}

pub fn end_capture(hstream: *mut stream::Stream) -> Result<*mut Graph, CUresult> {
    let capture = {
        let mut guard = CAPTURE.lock().unwrap_or_else(|poison| poison.into_inner());
        match guard.as_ref() {
            // Ending a capture that was never begun, or ending it from another
            // stream, is CUDA's UNMATCHED.
            None => return Err(CUresult::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED),
            Some(capture) if capture.stream != stream_key(hstream) => {
                return Err(CUresult::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED)
            }
            Some(_) => guard.take().unwrap(),
        }
    };
    ACTIVE.store(false, Ordering::Relaxed);
    let ops = if capture.ops.is_empty() {
        vec![Op::Empty]
    } else {
        capture.ops
    };
    let graph = Box::new(Graph::new(GraphData { ops, closed: false }));
    Ok(Box::into_raw(graph))
}

pub fn get_capture_info(
    hstream: *mut stream::Stream,
    status: *mut CUstreamCaptureStatus,
    id: *mut u64,
) -> Result<(), CUresult> {
    let guard = CAPTURE.lock().unwrap_or_else(|poison| poison.into_inner());
    let (state, capture_id) = match guard.as_ref() {
        Some(capture) if capture.stream == stream_key(hstream) => (
            CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
            capture.id,
        ),
        _ => (CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE, 0),
    };
    drop(guard);
    if !status.is_null() {
        unsafe { *status = state };
    }
    if !id.is_null() {
        unsafe { *id = capture_id };
    }
    Ok(())
}

pub fn exchange_mode(mode: *mut CUstreamCaptureMode) -> Result<(), CUresult> {
    if mode.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mut guard = CAPTURE.lock().unwrap_or_else(|poison| poison.into_inner());
    let previous = match guard.as_mut() {
        Some(capture) => {
            let previous = capture.mode;
            capture.mode = unsafe { *mode };
            previous
        }
        None => CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
    };
    unsafe { *mode = previous };
    Ok(())
}

// ---------------------------------------------------------------------------
// Graph and executable
// ---------------------------------------------------------------------------

pub fn create(ph_graph: *mut *mut Graph) -> Result<(), CUresult> {
    if ph_graph.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let graph = Box::new(Graph::new(GraphData {
        ops: Vec::new(),
        closed: false,
    }));
    unsafe { *ph_graph = Box::into_raw(graph) };
    Ok(())
}

pub fn destroy(graph: *mut Graph) -> Result<(), CUresult> {
    if graph.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    Graph::destroy_impl(graph)
}

pub fn exec_destroy(exec: *mut GraphExec) -> Result<(), CUresult> {
    if exec.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    GraphExec::destroy_impl(exec)
}

/// Build an executable from a captured graph.
///
/// The deep copy is not redundant: CUDA lets an application destroy the graph
/// and keep launching the executable, and the executable owns the device buffers
/// the recorded argument blocks live in.
pub fn instantiate(
    ph_exec: *mut *mut GraphExec,
    graph: *mut Graph,
    flags: u64,
) -> Result<(), CUresult> {
    if ph_exec.is_null() || graph.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let graph_data = unsafe { &*graph }
        .as_result()
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?;
    let mut ops = Vec::with_capacity(graph_data.ops.len());
    for op in graph_data.ops.iter() {
        ops.push(match op {
            Op::Launch(launch) => {
                let mut copy = Launch {
                    func: launch.func,
                    name: launch.name.clone(),
                    grid: launch.grid,
                    block: launch.block,
                    args: launch.args.clone(),
                    param_bytes: launch.param_bytes.clone(),
                    param_device: 0,
                    shared_window: launch.shared_window,
                    shared_arg_index: launch.shared_arg_index,
                };
                // The compiler refuses argument lists larger than 2048 bytes,
                // so such a kernel reads them from a block. The block is device
                // memory and belongs to the executable, which keeps it valid for
                // every replay; capturing the bytes once is exactly CUDA's
                // by-value contract.
                if let Some(bytes) = copy.param_bytes.as_ref() {
                    let device = super::memory::alloc_v2_raw(bytes.len())?;
                    super::memory::copy_v2(
                        device,
                        bytes.as_ptr() as *const c_void,
                        bytes.len(),
                    )?;
                    copy.param_device = device as usize;
                }
                Op::Launch(Box::new(copy))
            }
            Op::Copy { dst, src, bytes } => Op::Copy {
                dst: *dst,
                src: *src,
                bytes: *bytes,
            },
            Op::Fill {
                dst,
                pattern,
                pattern_size,
                bytes,
            } => Op::Fill {
                dst: *dst,
                pattern: *pattern,
                pattern_size: *pattern_size,
                bytes: *bytes,
            },
            Op::EventRecord { event } => Op::EventRecord { event: *event },
            Op::EventWait { event } => Op::EventWait { event: *event },
            Op::Empty => Op::Empty,
        });
    }
    let exec = Box::new(GraphExec::new(GraphExecData {
        ops,
        flags,
        closed: false,
    }));
    unsafe { *ph_exec = Box::into_raw(exec) };
    Ok(())
}

pub fn exec_flags(exec: *mut GraphExec, flags: *mut u64) -> Result<(), CUresult> {
    if exec.is_null() || flags.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    let data = unsafe { &*exec }
        .as_result()
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?;
    unsafe { *flags = data.flags };
    Ok(())
}

/// Replay an executable onto a stream.
///
/// Each recorded operation goes through the same enqueue the direct path uses, so
/// a replayed graph leaves the stream in exactly the state N direct operations
/// would have: one marker event per operation, in order, and the stream's
/// synchronise behaves the same. That costs the per-launch overhead again, which
/// is the price of a re-append design and is why this is a compatibility feature
/// rather than a speed one.
pub fn launch(exec: *mut GraphExec, hstream: *mut stream::Stream) -> Result<(), CUresult> {
    if exec.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    if capturing_any() {
        // A graph launch inside a capture would have to become a child-graph
        // node. Refusing is honest; recording it as its operations would flatten
        // the graph and lose the pool that keeps its addresses static.
        return Err(CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    }
    let exec_data = unsafe { &*exec }
        .as_result()
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?;
    for op in exec_data.ops.iter() {
        match op {
            Op::Empty => {}
            Op::Launch(_) | Op::Copy { .. } | Op::Fill { .. } => {
                super::GlobalState::lock_enqueue(hstream, |cmd_list, signal, wait| {
                    emit(op, cmd_list, signal, wait)
                })?;
            }
            // A barrier carries no work of its own. Appending one through the
            // enqueue would burn an event for nothing, so the two are emitted
            // directly, in order, on the stream's list.
            Op::EventRecord { event } | Op::EventWait { event } => {
                super::GlobalState::lock_stream(hstream, |stream_data| {
                    let l0_event = unsafe { &*(*event as *mut event::Event) }
                        .as_result()
                        .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?
                        .l0_event()
                        .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
                    let cmd_list = unsafe { super::transmute_lifetime(&stream_data.cmd_list) };
                    let empty: [&l0::Event; 0] = [];
                    unsafe {
                        match op {
                            Op::EventRecord { .. } => {
                                let _ = l0_event.host_reset();
                                cmd_list
                                    .append_barrier(Some(l0_event), &empty)
                                    .map_err(CUresult::from)?;
                            }
                            _ => cmd_list
                                .append_barrier(None, &[l0_event])
                                .map_err(CUresult::from)?,
                        }
                    }
                    Ok::<(), CUresult>(())
                })?;
            }
        }
    }
    Ok(())
}

fn emit(
    op: &Op,
    cmd_list: &l0::CommandList,
    signal: &l0::Event<'static>,
    wait: &[&l0::Event<'static>],
) -> Result<(), CUresult> {
    match op {
        Op::Launch(launch) => {
            let func = unsafe { &mut *(launch.func as *mut function::Function) }
                .as_result_mut()
                .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            if launch.param_bytes.is_some() {
                if launch.param_device == 0 {
                    return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
                }
                let block = launch.param_device as *mut c_void;
                unsafe {
                    func.base.set_arg_raw(
                        0,
                        std::mem::size_of::<*mut c_void>(),
                        &block as *const _ as *const c_void,
                    )?
                };
            } else {
                for (index, bytes) in launch.args.iter().enumerate() {
                    if bytes.is_empty() {
                        continue;
                    }
                    unsafe {
                        func.base.set_arg_raw(
                            index as u32,
                            bytes.len(),
                            bytes.as_ptr() as *const c_void,
                        )?
                    };
                }
            }
            if let Some(window) = launch.shared_window {
                unsafe { func.base.set_arg_raw(launch.shared_arg_index, window, ptr::null())? };
            }
            func.base
                .set_group_size(launch.block[0], launch.block[1], launch.block[2])
                .map_err(CUresult::from)?;
            func.legacy_args.reset();
            unsafe {
                cmd_list
                    .append_launch_kernel(&mut func.base, &launch.grid, Some(signal), wait)
                    .map_err(CUresult::from)?;
            }
            Ok(())
        }
        Op::Copy { dst, src, bytes } => unsafe {
            cmd_list
                .append_memory_copy_raw(
                    *dst as *mut c_void,
                    *src as *const c_void,
                    *bytes,
                    Some(signal),
                    wait,
                )
                .map_err(CUresult::from)
        },
        Op::Fill {
            dst,
            pattern,
            pattern_size,
            bytes,
        } => unsafe {
            cmd_list
                .append_memory_fill_raw(
                    *dst as *mut c_void,
                    pattern.as_ptr() as *mut c_void,
                    *pattern_size,
                    *bytes,
                    Some(signal),
                    wait,
                )
                .map_err(CUresult::from)
        },
        _ => Ok(()),
    }
}

/// How many recorded operations a graph holds.
///
/// `cuGraphGetNodes` answers with this: torch's `capture_end` counts the nodes of
/// the graph it just captured and warns when the count is zero, which is how a
/// capture that silently recorded nothing is supposed to be noticed.
pub fn node_count(graph: *mut Graph) -> Result<usize, CUresult> {
    if graph.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    let data = unsafe { &*graph }
        .as_result()
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_HANDLE)?;
    Ok(data.ops.len())
}

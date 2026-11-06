use super::{stream, CUresult, GlobalState};
use std::{
    collections::HashMap,
    ffi::c_void,
    mem, ptr,
    sync::{Mutex, OnceLock},
};

/// Allocates device memory and returns the raw pointer (no dptr out-param).
pub fn alloc_v2_raw(bytesize: usize) -> Result<*mut c_void, CUresult> {
    let alloc_ptr =
GlobalState::lock_current_context_unchecked(|ctx| {
let ctx_ptr = ctx as *mut _;
let ctx_data = ctx.as_result_mut()?;
let dev = unsafe { &mut *ctx_data.device };
let alloc_ptr = dev.l0_context.mem_alloc_device(bytesize, 0, dev.base)?;
ctx_data.pointer_registry.register(super::pointer::Allocation {
base: alloc_ptr,
size: bytesize,
is_host: false,
context: ctx_ptr,
});
Ok::<_, CUresult>(alloc_ptr)
})??;
// A fresh allocation is cleared: applications rely on cudaMalloc memory
// reading as zeros. ZLUDA_NO_ALLOC_ZERO=1 opts out.
if std::env::var_os("ZLUDA_NO_ALLOC_ZERO").is_none() && bytesize > 0 {
    set_d8_v2(alloc_ptr, 0, bytesize, stream::CU_STREAM_LEGACY)?;
}
Ok(alloc_ptr)
}

pub fn alloc_v2(dptr: *mut *mut c_void, bytesize: usize) -> Result<(), CUresult> {
    let ptr = alloc_v2_raw(bytesize)?;
    unsafe { *dptr = ptr };
    Ok(())
}

pub fn copy_v2(dst: *mut c_void, src: *const c_void, bytesize: usize) -> Result<(), CUresult> {
// A zero-byte copy is a no-op, and one that reaches the copy engine hangs the
// host (see copy_async).
if bytesize == 0 {
    return Ok(());
}
// A blocking copy runs on the legacy stream, which CUDA orders against every
// other blocking stream. The immediate command lists are independent queues,
// so the ordering has to be made here.
GlobalState::synchronize_context_streams()?;
// A sync copy must block until it retires, or a following readback races it.
GlobalState::lock_enqueue(stream::CU_STREAM_LEGACY, |cmd_list, signal, wait| {
    unsafe { cmd_list.append_memory_copy_raw(dst, src, bytesize, Some(signal), wait)? };
    Ok(())
})?;
stream::synchronize(stream::CU_STREAM_LEGACY)?;
Ok(())
}

/// A pinned staging buffer taken from the pool. One copy holds a buffer at a
/// time; the driver faults when a copy engine touches pageable host memory.
/// Buffers are never freed, only returned once their copy has retired.
struct Staging {
    class: usize,
    ptr: *mut c_void,
}

impl Drop for Staging {
    fn drop(&mut self) {
        let mut pool = staging_pool().lock().unwrap();
        pool.entry(self.class).or_default().push(SendPtr(self.ptr));
    }
}

/// The size class a buffer of `bytesize` is rounded up to. Classes are powers
/// of two from 1 MiB up: keying by the exact size pins a buffer per size until
/// host memory runs out.
fn staging_class(bytesize: usize) -> usize {
    bytesize.max(1 << 20).next_power_of_two().max(1 << 20)
}

fn staging_pool() -> &'static Mutex<HashMap<usize, Vec<SendPtr>>> {
    static POOL: OnceLock<Mutex<HashMap<usize, Vec<SendPtr>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Take a pinned staging buffer that fits `bytesize`.
fn take_staging(bytesize: usize) -> Result<Staging, CUresult> {
    let class = staging_class(bytesize);
    let free = staging_pool()
        .lock()
        .unwrap()
        .entry(class)
        .or_default()
        .pop();
    if let Some(SendPtr(ptr)) = free {
        return Ok(Staging { class, ptr });
    }
    let ptr = GlobalState::lock_current_context_unchecked(|ctx| {
        let ctx_data = ctx.as_result_mut()?;
        let dev = unsafe { &mut *ctx_data.device };
        Ok::<_, CUresult>(dev.l0_context.mem_alloc_host(class, 0)?)
    })??;
    Ok(Staging { class, ptr })
}

/// True if `p` is NOT inside a tracked pinned-host allocation. Callers pass
/// direction-appropriate host-side operands only.
fn host_untracked(p: *const c_void) -> bool {
    GlobalState::lock_current_context_unchecked(|ctx| {
        let ctx_data = ctx.as_result_mut()?;
        Ok::<_, CUresult>(
            ctx_data
                .pointer_registry
                .find(p as *mut c_void)
                .map(|a| !a.is_host)
                .unwrap_or(true),
        )
    })
    .map(|b| b.unwrap_or(true))
    .unwrap_or(true)
}

struct SendPtr(*mut c_void);
unsafe impl Send for SendPtr {}

/// Async device-to-device copy on a stream (cuMemcpyAsync / cuMemcpyDtoDAsync_v2).
pub fn copy_async(
    dst: *mut c_void,
    src: *const c_void,
    bytesize: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    // A zero-byte copy is a no-op in CUDA. Handing it to the copy engine blocks
    // the host here (measured: a zero-length cuMemcpyDtoH hangs until the
    // caller's timeout), and torch issues one for every empty tensor.
    if bytesize == 0 {
        return Ok(());
    }
    GlobalState::lock_enqueue(hstream, |cmd_list, signal, wait| {
        unsafe { cmd_list.append_memory_copy_raw(dst, src, bytesize, Some(signal), wait)? };
        Ok(())
    })
}

/// cuMemcpyHtoDAsync_v2.
pub fn copy_htod_async(
    dst: *mut c_void,
    src: *const c_void,
    bytesize: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    if host_untracked(src) {
        // Unpinned host source: bounce through the pinned staging pool, since
        // the engine faults on pageable pages. Synchronize before returning so
        // the staging buffer is safe to reuse.
        let stage = take_staging(bytesize)?;
        unsafe { ptr::copy_nonoverlapping(src.cast::<u8>(), stage.ptr.cast::<u8>(), bytesize) };
        copy_async(dst, stage.ptr, bytesize, hstream)?;
        return GlobalState::lock_stream(hstream, |sd| {
            sd.synchronize().map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)
        })
        ?;
    }
    copy_async(dst, src, bytesize, hstream)
}

/// cuMemcpyDtoHAsync_v2.
pub fn copy_dtoh_async(
    dst: *mut c_void,
    src: *const c_void,
    bytesize: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    if bytesize == 0 {
        return Ok(());
    }
    if host_untracked(dst) {
        // Unpinned host destination: bounce through cached pinned staging.
        let stage = take_staging(bytesize)?;
        copy_async(stage.ptr, src, bytesize, hstream)?;
        GlobalState::lock_stream(hstream, |sd| {
            sd.synchronize().map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)
        })??;
        unsafe { ptr::copy_nonoverlapping(stage.ptr.cast::<u8>(), dst.cast::<u8>(), bytesize) };
        return Ok(());
    }
    copy_async(dst, src, bytesize, hstream)
}

/// cuMemcpyDtoDAsync_v2.
pub fn copy_dtod_async(
    dst: *mut c_void,
    src: *const c_void,
    bytesize: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    copy_async(dst, src, bytesize, hstream)
}

/// Sync HtoD on the legacy stream; bounces unpinned host sources.
pub fn copy_htod(
    dst: *mut c_void,
    src: *const c_void,
    bytesize: usize,
) -> Result<(), CUresult> {
    // Work queued on the application's streams has to finish first, or the
    // copy reads a buffer a kernel still has to write.
    GlobalState::synchronize_context_streams()?;
    if host_untracked(src) {
        // Unpinned host source: bounce through cached pinned staging.
        let stage = take_staging(bytesize)?;
        unsafe { ptr::copy_nonoverlapping(src.cast::<u8>(), stage.ptr.cast::<u8>(), bytesize) };
        copy_async(dst, stage.ptr, bytesize, stream::CU_STREAM_LEGACY)?;
    } else {
        copy_async(dst, src, bytesize, stream::CU_STREAM_LEGACY)?;
    }
    stream::synchronize(stream::CU_STREAM_LEGACY)
}

/// Sync DtoH on the legacy stream; bounces unpinned host destinations.
pub fn copy_dtoh(
    dst: *mut c_void,
    src: *const c_void,
    bytesize: usize,
) -> Result<(), CUresult> {
    GlobalState::synchronize_context_streams()?;
    if host_untracked(dst) {
        // Unpinned host destination: bounce through cached pinned staging.
        let stage = take_staging(bytesize)?;
        copy_async(stage.ptr, src, bytesize, stream::CU_STREAM_LEGACY)?;
        stream::synchronize(stream::CU_STREAM_LEGACY)?;
        unsafe { ptr::copy_nonoverlapping(stage.ptr.cast::<u8>(), dst.cast::<u8>(), bytesize) };
        Ok(())
    } else {
        copy_async(dst, src, bytesize, stream::CU_STREAM_LEGACY)?;
        stream::synchronize(stream::CU_STREAM_LEGACY)
    }
}
pub fn copy_dtod(dst: *mut c_void, src: *const c_void, bytesize: usize) -> Result<(), CUresult> {
    copy_async(dst, src, bytesize, stream::CU_STREAM_LEGACY)?;
    stream::synchronize(stream::CU_STREAM_LEGACY)
}

    pub fn free_v2(ptr: *mut c_void) -> Result<(), CUresult> {
    // cuMemFree(NULL) is a no-op on real CUDA; cuBLAS sometimes frees NULL.
    if ptr.is_null() {
        return Ok(());
    }
    // ZLUDA_LEAK_FREE is a diagnostic: freed memory is never handed back to
    // the driver, so no later allocation can land on it.
    if std::env::var_os("ZLUDA_LEAK_FREE").is_some() {
        return Ok(());
    }
    // Unmapping memory that queued work still references faults the engine and
    // wedges every later immediate list, so drain the context first.
    GlobalState::synchronize_context_streams()?;
    GlobalState::lock_current_context(|ctx| {
        let dev = unsafe { &mut *ctx.device };
        ctx.pointer_registry.unregister(ptr);
        Ok::<_, CUresult>(dev.l0_context.mem_free(ptr)?)
    })
    .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?
}

/// cuMemPoolCreate: return an opaque dummy handle. ZLUDA allocs bypass pools.
pub fn pool_create(pool: *mut crate::cuda::CUmemoryPool) -> Result<(), CUresult> {
    if pool == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    static DUMMY_POOL: crate::cuda::CUmemPoolHandle_st = unsafe { mem::zeroed() };
    unsafe { *pool = &DUMMY_POOL as *const _ as *mut crate::cuda::CUmemPoolHandle_st };
    Ok(())
}

pub fn pool_destroy(_pool: crate::cuda::CUmemoryPool) -> Result<(), CUresult> {
    Ok(())
}

pub fn get_info_v2(free: *mut usize, total: *mut usize) -> Result<(), CUresult> {
    if free == ptr::null_mut() || total == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let (free_bytes, total_bytes) = GlobalState::lock_current_context(|ctx| {
        let dev = unsafe { &mut *ctx.device };
        let mem_props = dev.get_memory_properties().map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
        let total_bytes = mem_props.iter().map(|p| p.totalSize).max().unwrap_or(0) as usize;
        // L0 does not expose free memory; report total as free so apps do not
        // refuse to run thinking the device is out of memory.
        Ok::<_, CUresult>((total_bytes, total_bytes))
    })??;
    unsafe {
        *free = free_bytes;
        *total = total_bytes;
    }
    Ok(())
}

pub fn alloc_host_v2(pp: *mut *mut c_void, bytesize: usize) -> Result<(), CUresult> {
    if pp == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let ptr = GlobalState::lock_current_context_unchecked(|ctx| {
        let ctx_ptr = ctx as *mut _;
        let ctx_data = ctx.as_result_mut()?;
        let dev = unsafe { &mut *ctx_data.device };
        let ptr = dev.l0_context.mem_alloc_host(bytesize, 0)?;
        ctx_data.pointer_registry.register(super::pointer::Allocation {
            base: ptr,
            size: bytesize,
            is_host: true,
            context: ctx_ptr,
        });
        Ok::<_, CUresult>(ptr)
    })??;
    unsafe { *pp = ptr };
    Ok(())
}

/// A pinned host allocation that is never freed.
///
/// Used by the parameter-buffer slots: the launch stages its arguments here and
/// the command list copies them to the device, and this driver cannot DMA from a
/// pageable buffer.
pub fn alloc_host_pinned(bytesize: usize) -> Result<*mut c_void, CUresult> {
    GlobalState::lock_current_context_unchecked(|ctx| {
        let ctx_ptr = ctx as *mut _;
        let ctx_data = ctx.as_result_mut()?;
        let dev = unsafe { &mut *ctx_data.device };
        let ptr = dev.l0_context.mem_alloc_host(bytesize, 0)?;
        ctx_data.pointer_registry.register(super::pointer::Allocation {
            base: ptr as *mut c_void,
            size: bytesize,
            is_host: true,
            context: ctx_ptr,
        });
        Ok::<_, CUresult>(ptr as *mut c_void)
    })?
}

pub fn host_alloc(
    pp: *mut *mut c_void,
    bytesize: usize,
    _flags: u32,
) -> Result<(), CUresult> {
    // L0 host allocations are directly accessible from the device, so the
    // CU_MEMHOSTALLOC_* flag combinations collapse to the same allocation.
    alloc_host_v2(pp, bytesize)
}

pub fn host_get_device_pointer_v2(
    pdptr: *mut *mut c_void,
    p: *mut c_void,
    _flags: u32,
) -> Result<(), CUresult> {
    if pdptr == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    // L0 host allocations are directly accessible from the device, so the
    // device pointer is the host pointer itself.
    unsafe { *pdptr = p };
    Ok(())
}

pub fn host_get_flags(pflags: *mut u32, _p: *mut c_void) -> Result<(), CUresult> {
    if pflags == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    unsafe { *pflags = 0 };
    Ok(())
}

pub fn free_host(p: *mut c_void) -> Result<(), CUresult> {
if p == ptr::null_mut() {
return Ok(());
}
// Real CUDA defers unmapping host-pinned pages until queued DMA touching them
// retires; L0 unmaps immediately, so drain the context first.
GlobalState::synchronize_context_streams()?;
GlobalState::lock_current_context(|ctx| {
        let dev = unsafe { &mut *ctx.device };
        ctx.pointer_registry.unregister(p);
Ok::<_, CUresult>(dev.l0_context.mem_free(p)?)
})
.map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?
}

/// A fill runs on the stream the caller named: two streams of this driver are
/// two independent command lists, so a fill on the wrong one has no ordering
/// relationship with the work that reads it. The pattern points at the
/// caller's stack, so the fill is waited for before returning.
fn set_pattern(
    dst: *mut c_void,
    pattern: *mut c_void,
    pattern_size: usize,
    size: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    if hstream == stream::CU_STREAM_LEGACY {
        // A fill on the legacy stream is ordered against every other stream.
        GlobalState::synchronize_context_streams()?;
    }
    GlobalState::lock_enqueue(hstream, |cmd_list, signal, wait| {
        unsafe {
            cmd_list.append_memory_fill_raw(dst, pattern, pattern_size, size, Some(signal), wait)
        }?;
        Ok(())
    })?;
    stream::synchronize(hstream)
}

pub(crate) fn set_d32_v2(
    dst: *mut c_void,
    mut ui: u32,
    n: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    set_pattern(
        dst,
        &mut ui as *mut _ as *mut _,
        mem::size_of::<u32>(),
        mem::size_of::<u32>() * n,
        hstream,
    )
}

pub(crate) fn set_d8_v2(
    dst: *mut c_void,
    mut uc: u8,
    n: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    set_pattern(
        dst,
        &mut uc as *mut _ as *mut _,
        mem::size_of::<u8>(),
        mem::size_of::<u8>() * n,
        hstream,
    )
}

pub(crate) fn set_d16_v2(
    dst: *mut c_void,
    mut us: u16,
    n: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    set_pattern(
        dst,
        &mut us as *mut _ as *mut _,
        mem::size_of::<u16>(),
        mem::size_of::<u16>() * n,
        hstream,
    )
}


fn src_ptr(mem_type: u32, host: *const c_void, dev: usize) -> *const c_void {
    // CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2
    if mem_type == 1 {
        host
    } else {
        dev as *const c_void
    }
}

fn dst_ptr(mem_type: u32, host: *mut c_void, dev: usize) -> *mut c_void {
    if mem_type == 1 {
        host
    } else {
        dev as *mut c_void
    }
}

/// cuMemcpy2D_v2 / cuMemcpy2DUnaligned_v2 via L0 region copy.
pub fn memcpy2d(
    src_x: usize,
    src_y: usize,
    src_type: u32,
    src_host: *const c_void,
    src_dev: usize,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: u32,
    dst_host: *mut c_void,
    dst_dev: usize,
    dst_pitch: usize,
    width: usize,
    height: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    let src_base = src_ptr(src_type, src_host, src_dev);
    let dst_base = dst_ptr(dst_type, dst_host, dst_dev);
    // Intel's L0 driver mishandles region copies whose destination pitch is
    // larger than the width (padded rows come out zeroed), so one contiguous
    // copy per row is issued instead.
    GlobalState::lock_enqueue(hstream, |cmd_list, signal, wait| {
        for row in 0..height {
            let src_row = unsafe { src_base.add((src_y + row) * src_pitch + src_x) };
            let dst_row = unsafe { dst_base.add((dst_y + row) * dst_pitch + dst_x) };
            let last = row == height - 1;
            unsafe {
                cmd_list.append_memory_copy_raw(
                    dst_row,
                    src_row,
                    width,
                    if last { Some(signal) } else { None },
                    wait,
                )?;
            }
        }
        Ok(())
    })?;
    // The sync variants must block, or a host readback races the last row.
    if hstream == stream::CU_STREAM_LEGACY || hstream == stream::CU_STREAM_PER_THREAD {
        stream::synchronize(hstream)?;
    }
    Ok(())
}

/// cuMemcpy3D_v2 via L0 region copy.
#[allow(clippy::too_many_arguments)]
pub fn memcpy3d(
    src_x: usize,
    src_y: usize,
    src_z: usize,
    src_type: u32,
    src_host: *const c_void,
    src_dev: usize,
    src_pitch: usize,
    src_height: usize,
    dst_x: usize,
    dst_y: usize,
    dst_z: usize,
    dst_type: u32,
    dst_host: *mut c_void,
    dst_dev: usize,
    dst_pitch: usize,
    dst_height: usize,
    width: usize,
    height: usize,
    depth: usize,
    hstream: *mut stream::Stream,
) -> Result<(), CUresult> {
    let src_base = src_ptr(src_type, src_host, src_dev);
    let dst_base = dst_ptr(dst_type, dst_host, dst_dev);
    // Per-row copies, for the reason given in memcpy2d.
    let src_slice = src_pitch * src_height;
    let dst_slice = dst_pitch * dst_height;
    GlobalState::lock_enqueue(hstream, |cmd_list, signal, wait| {
        let mut last_signal = None;
        for z in 0..depth {
            for row in 0..height {
                let src_row = unsafe {
                    src_base.add((src_z + z) * src_slice + (src_y + row) * src_pitch + src_x)
                };
                let dst_row = unsafe {
                    dst_base.add((dst_z + z) * dst_slice + (dst_y + row) * dst_pitch + dst_x)
                };
                let is_last = z == depth - 1 && row == height - 1;
                unsafe {
                    cmd_list.append_memory_copy_raw(
                        dst_row,
                        src_row,
                        width,
                        if is_last { Some(signal) } else { None },
                        wait,
                    )?;
                }
                if is_last {
                    last_signal = Some(());
                }
            }
        }
        debug_assert!(last_signal.is_some(), "depth*height must be > 0");
        Ok(())
    })?;
    if hstream == stream::CU_STREAM_LEGACY || hstream == stream::CU_STREAM_PER_THREAD {
        stream::synchronize(hstream)?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::super::test::CudaDriverFns;
    use super::super::CUresult;
    use std::ptr;

    cuda_driver_test!(alloc_without_ctx);

    fn alloc_without_ctx<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut mem = ptr::null_mut();
        assert_eq!(
            T::cuMemAlloc_v2(&mut mem, std::mem::size_of::<usize>()),
            CUresult::CUDA_ERROR_INVALID_CONTEXT
        );
        assert_eq!(mem, ptr::null_mut());
    }

    cuda_driver_test!(alloc_with_ctx);

    fn alloc_with_ctx<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx, 0, 0), CUresult::CUDA_SUCCESS);
        let mut mem = ptr::null_mut();
        assert_eq!(
            T::cuMemAlloc_v2(&mut mem, std::mem::size_of::<usize>()),
            CUresult::CUDA_SUCCESS
        );
        assert_ne!(mem, ptr::null_mut());
        assert_eq!(T::cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
    }

    cuda_driver_test!(free_without_ctx);

    fn free_without_ctx<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut ctx = ptr::null_mut();
        assert_eq!(T::cuCtxCreate_v2(&mut ctx, 0, 0), CUresult::CUDA_SUCCESS);
        let mut mem = ptr::null_mut();
        assert_eq!(
            T::cuMemAlloc_v2(&mut mem, std::mem::size_of::<usize>()),
            CUresult::CUDA_SUCCESS
        );
        assert_ne!(mem, ptr::null_mut());
        assert_eq!(T::cuCtxDestroy_v2(ctx), CUresult::CUDA_SUCCESS);
        assert_eq!(T::cuMemFree_v2(mem), CUresult::CUDA_ERROR_INVALID_VALUE);
    }
}

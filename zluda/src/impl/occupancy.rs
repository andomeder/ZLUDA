//! Occupancy queries (`cuOccupancy*`).
//!
//! Applications use these to pick a launch configuration, so they answer what a
//! real device would from the Level Zero limits. The register file is not
//! reported per kernel, so the limit that comes from it is not modelled: a
//! kernel short of registers gets a group size the driver refuses at launch,
//! which is loud rather than silent.

use super::{device, function, CUresult, GlobalState};
use crate::cuda::CUdevice_attribute;
use std::ffi::c_int;

/// The occupancy limits of one multiprocessor.
struct Limits {
    max_threads: i32,
    max_blocks: i32,
    max_shared: i32,
}

fn device_attribute(attrib: CUdevice_attribute, dev_idx: device::Index) -> Result<i32, CUresult> {
    let mut value: i32 = 0;
    device::get_attribute(&mut value, attrib, dev_idx)?;
    Ok(value)
}

fn limits(dev_idx: device::Index) -> Result<Limits, CUresult> {
    Ok(Limits {
        max_threads: device_attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            dev_idx,
        )?,
        max_blocks: device_attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
            dev_idx,
        )?,
        max_shared: device_attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            dev_idx,
        )?,
    })
}

/// The number of blocks of `block_size` threads that fit on one multiprocessor.
fn blocks_per_multiprocessor(
    limits: &Limits,
    block_size: i32,
    shared_bytes: usize,
) -> i32 {
    if block_size <= 0 {
        return 0;
    }
    let mut blocks = limits.max_threads / block_size;
    if shared_bytes > 0 {
        blocks = blocks.min((limits.max_shared as usize / shared_bytes) as i32);
    }
    blocks.min(limits.max_blocks).max(0)
}

fn function_attribute(
    attrib: crate::cuda::CUfunction_attribute,
    func: *mut function::Function,
) -> Result<i32, CUresult> {
    let mut value: i32 = 0;
    function::get_attribute(&mut value, attrib, func)?;
    Ok(value)
}

/// The static shared memory of the kernel and the largest block it allows.
fn function_resources(func: *mut function::Function) -> Result<(i32, i32), CUresult> {
    Ok((
        function_attribute(
            crate::cuda::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            func,
        )?,
        function_attribute(
            crate::cuda::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            func,
        )?,
    ))
}

fn current_device() -> Result<device::Index, CUresult> {
    GlobalState::lock_current_context(|ctx| Ok::<_, CUresult>(unsafe { &*ctx.device }.index))?
}

pub fn max_active_blocks_per_multiprocessor(
    num_blocks: *mut c_int,
    func: *mut function::Function,
    block_size: c_int,
    dynamic_smem: usize,
) -> Result<(), CUresult> {
    if num_blocks.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    if func.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }
    let dev_idx = current_device()?;
    let limits = limits(dev_idx)?;
    let (static_smem, kernel_max_threads) = function_resources(func)?;
    if block_size > kernel_max_threads || block_size <= 0 {
        // A block that is bigger than the kernel allows cannot be resident.
        unsafe { *num_blocks = 0 };
        return Ok(());
    }
    let shared = static_smem as usize + dynamic_smem;
    unsafe { *num_blocks = blocks_per_multiprocessor(&limits, block_size, shared) };
    Ok(())
}

pub fn max_active_blocks_per_multiprocessor_with_flags(
    num_blocks: *mut c_int,
    func: *mut function::Function,
    block_size: c_int,
    dynamic_smem: usize,
    _flags: u32,
) -> Result<(), CUresult> {
    // The flags only pin the cache configuration, which a kernel here does not
    // choose.
    max_active_blocks_per_multiprocessor(num_blocks, func, block_size, dynamic_smem)
}

/// The block size that fills a multiprocessor best, and the grid that fills
/// the device with it.
pub fn max_potential_block_size(
    min_grid_size: *mut c_int,
    block_size_out: *mut c_int,
    func: *mut function::Function,
    block_size_to_smem: crate::cuda::CUoccupancyB2DSize,
    dynamic_smem: usize,
    block_size_limit: c_int,
) -> Result<(), CUresult> {
    if min_grid_size.is_null() || block_size_out.is_null() || func.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let dev_idx = current_device()?;
    let limits = limits(dev_idx)?;
    let (static_smem, kernel_max_threads) = function_resources(func)?;
    let multiprocessors = device_attribute(
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        dev_idx,
    )?;
    let device_max_threads = device_attribute(
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        dev_idx,
    )?;
    let mut ceiling = kernel_max_threads.min(device_max_threads).max(32);
    if block_size_limit > 0 {
        ceiling = ceiling.min(block_size_limit);
    }
    // Search in warp sized steps for the size that puts the most threads on a
    // multiprocessor; the smallest size wins a tie.
    let mut best_size = 32;
    let mut best_threads = 0;
    let mut best_blocks = 0;
    let mut size = 32;
    while size <= ceiling {
        let extra = match block_size_to_smem {
            // The callback is part of the driver API and has to be called.
            Some(f) => unsafe { f(size as c_int) },
            None => 0,
        };
        let shared = static_smem as usize + dynamic_smem + extra;
        let blocks = blocks_per_multiprocessor(&limits, size, shared);
        let threads = blocks * size;
        if threads > best_threads {
            best_threads = threads;
            best_size = size;
            best_blocks = blocks;
        }
        size += 32;
    }
    unsafe {
        *block_size_out = best_size;
        *min_grid_size = (best_blocks * multiprocessors).max(multiprocessors);
    }
    Ok(())
}

pub fn available_dynamic_smem_per_block(
    dynamic_smem_out: *mut usize,
    func: *mut function::Function,
    num_blocks: c_int,
    _block_size: c_int,
) -> Result<(), CUresult> {
    if dynamic_smem_out.is_null() || func.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    if num_blocks <= 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let dev_idx = current_device()?;
    let limits = limits(dev_idx)?;
    let (static_smem, _) = function_resources(func)?;
    let per_block = limits.max_shared as usize / num_blocks as usize;
    unsafe { *dynamic_smem_out = per_block.saturating_sub(static_smem as usize) };
    Ok(())
}

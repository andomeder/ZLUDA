use super::{context, GlobalState, CUresult};
use std::{collections::BTreeMap, ffi::c_void, os::raw::c_int, ptr};

/// Metadata for a single device/host allocation tracked for
/// cuPointerGetAttribute / cuPointerGetAttributes.
#[derive(Debug, Clone, Copy)]
pub struct Allocation {
    pub base: *mut c_void,
    pub size: usize,
    /// True for cuMemAllocHost_v2 / cuMemHostAlloc (device-accessible host mem).
    pub is_host: bool,
    pub context: *mut context::Context,
}

/// Tracks allocations per context so cuPointerGetAttribute can classify an
/// arbitrary pointer (base or interior) as device/host and report its range.
/// The registry lives in ContextData and is only accessed under the global
/// state lock, so no locking of its own is needed.
#[derive(Default)]
pub struct PointerRegistry {
    /// Sorted by base address for range lookups.
    ranges: BTreeMap<usize, Allocation>,
}

impl PointerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, alloc: Allocation) {
        self.ranges.insert(alloc.base as usize, alloc);
    }

    /// Removes an allocation. Returns the removed metadata.
    pub fn unregister(&mut self, ptr: *mut c_void) -> Option<Allocation> {
        self.ranges.remove(&(ptr as usize))
    }

    /// Finds the allocation containing `ptr` (base or interior address).
    pub fn find(&self, ptr: *mut c_void) -> Option<&Allocation> {
        let addr = ptr as usize;
        self.ranges
            .range(..=addr)
            .next_back()
            .filter(|(_, alloc)| addr < alloc.base as usize + alloc.size)
            .map(|(_, alloc)| alloc)
    }
}

/// Write one attribute of `alloc` into `data`.
pub fn write_attribute(
    data: *mut c_void,
    attribute: u32,
    alloc: &Allocation,
    dev_ordinal: i32,
) -> Result<(), CUresult> {
    if data == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    // CUpointer_attribute enum values (see cuda.rs).
    const CONTEXT: u32 = 1;
    const MEMORY_TYPE: u32 = 2;
    const DEVICE_POINTER: u32 = 3;
    const HOST_POINTER: u32 = 4;
    const BUFFER_ID: u32 = 7;
    const IS_MANAGED: u32 = 8;
    const DEVICE_ORDINAL: u32 = 9;
    const RANGE_START_ADDR: u32 = 11;
    const RANGE_SIZE: u32 = 12;
    const MAPPED: u32 = 13;
    match attribute {
        // CU_POINTER_ATTRIBUTE_CONTEXT: the context that owns the pointer.
        CONTEXT => unsafe {
            *(data as *mut *mut c_void) = alloc.context as *mut c_void;
        },
        // CU_POINTER_ATTRIBUTE_MEMORY_TYPE: CU_MEMORYTYPE_DEVICE=2 / CU_MEMORYTYPE_HOST=1.
        MEMORY_TYPE => unsafe {
            *(data as *mut c_int) = if alloc.is_host { 1 } else { 2 };
        },
        DEVICE_POINTER => unsafe {
            *(data as *mut *mut c_void) = alloc.base;
        },
        HOST_POINTER => unsafe {
            *(data as *mut *mut c_void) = if alloc.is_host { alloc.base } else { ptr::null_mut() };
        },
        IS_MANAGED => unsafe {
            *(data as *mut c_int) = 0;
        },
        DEVICE_ORDINAL => unsafe {
            *(data as *mut c_int) = dev_ordinal;
        },
        RANGE_START_ADDR => unsafe {
            *(data as *mut *mut c_void) = alloc.base;
        },
        RANGE_SIZE => unsafe {
            *(data as *mut usize) = alloc.size;
        },
        MAPPED => unsafe {
            // Managed-memory mapped flag; host-visible device mem maps to true.
            *(data as *mut c_int) = if alloc.is_host { 1 } else { 0 };
        },
        BUFFER_ID => unsafe {
            // Synthetic buffer id: the base address (stable, unique).
            *(data as *mut u64) = alloc.base as u64;
        },
        // Unsupported attributes return INVALID_VALUE like real CUDA for
        // attributes that don't apply to this pointer kind.
        _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
    }
    Ok(())
}

/// cuPointerGetAttribute: query one attribute of a pointer.
pub fn get_attribute(data: *mut c_void, attribute: u32, ptr: *mut c_void) -> Result<(), CUresult> {
let dev_ordinal = GlobalState::lock_current_context(|ctx| unsafe { &*ctx.device }.index.0)?;
GlobalState::lock_current_context(|ctx| {
let alloc = ctx
.pointer_registry
            .find(ptr)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        write_attribute(data, attribute, alloc, dev_ordinal)
})??;
Ok(())
}

/// cuPointerGetAttributes: query multiple attributes of a pointer.
pub fn get_attributes(
    num_attributes: u32,
    attributes: *const u32,
    data: *mut *mut c_void,
    ptr: *mut c_void,
) -> Result<(), CUresult> {
    if num_attributes == 0 {
        return Ok(());
    }
    if attributes == ptr::null_mut() || data == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let dev_ordinal = GlobalState::lock_current_context(|ctx| unsafe { &*ctx.device }.index.0)?;
    GlobalState::lock_current_context(|ctx| {
        let alloc = ctx
            .pointer_registry
            .find(ptr)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        for i in 0..num_attributes as usize {
            let attr = unsafe { *attributes.add(i) };
            let data_i = unsafe { *data.add(i) };
            write_attribute(data_i, attr, alloc, dev_ordinal)?;
        }
        Ok::<(), CUresult>(())
    })??;
    Ok(())
}


use super::{context, transmute_lifetime, transmute_lifetime_mut, CUresult, GlobalState};
use crate::cuda;
use cuda::{CUdevice_attribute, CUuuid_st};
use std::{
    cmp, mem,
    ffi::c_void,
    os::raw::{c_char, c_int, c_uint},
    ptr,
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
};

const PROJECT_URL_SUFFIX_SHORT: &'static str = " [ZLUDA]";
const PROJECT_URL_SUFFIX_LONG: &'static str = " [github.com/vosen/ZLUDA]";

#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Index(pub c_int);

pub struct Device {
    pub index: Index,
    pub base: l0::Device,
    pub default_queue: l0::CommandQueue<'static>,
    pub l0_context: l0::Context,
    pub primary_context: context::Context,
    pub device_event_pool: DynamicEventPool,
    pub host_event_pool: DynamicEventPool,
    properties: Option<Box<l0::sys::ze_device_properties_t>>,
    image_properties: Option<Box<l0::sys::ze_device_image_properties_t>>,
    memory_properties: Option<Vec<l0::sys::ze_device_memory_properties_t>>,
    compute_properties: Option<Box<l0::sys::ze_device_compute_properties_t>>,
    pub primary_ctx_retained: AtomicBool,
}

unsafe impl Send for Device {}

impl Device {
    // Unsafe: primary_context is filled in by late_init, and the lifetimes are
    // transmuted.
    unsafe fn new(drv: &l0::Driver, l0_dev: l0::Device, idx: usize) -> Result<Self, CUresult> {
        let ctx = l0::Context::new(*drv, Some(&[l0_dev]))?;
        let queue = l0::CommandQueue::new(mem::transmute(&ctx), l0_dev)?;
        let mut host_event_pool = DynamicEventPool::new(
            l0_dev,
            transmute_lifetime(&ctx),
            l0::sys::ze_event_pool_flags_t::ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
            // GPU queues signal these events and the host waits on them.
            l0::sys::ze_event_scope_flags_t::ZE_EVENT_SCOPE_FLAG_HOST,
        )?;
        let host_event =
            transmute_lifetime_mut(&mut host_event_pool).get(l0_dev, transmute_lifetime(&ctx))?;
        let primary_context = context::Context::new(context::ContextData::new(
            transmute_lifetime(&ctx),
            l0_dev,
            0,
            true,
            host_event,
            ptr::null_mut(),
        )?);
        let device_event_pool = DynamicEventPool::new(
            l0_dev,
            transmute_lifetime(&ctx),
            l0::sys::ze_event_pool_flags_t(0),
            l0::sys::ze_event_scope_flags_t(0),
        )?;
        Ok(Self {
            index: Index(idx as c_int),
            base: l0_dev,
            default_queue: queue,
            l0_context: ctx,
            primary_context: primary_context,
            device_event_pool,
            host_event_pool,
            properties: None,
            image_properties: None,
            memory_properties: None,
            compute_properties: None,
            primary_ctx_retained: AtomicBool::new(false),
        })
    }

    fn get_properties<'a>(&'a mut self) -> l0::Result<&'a l0::sys::ze_device_properties_t> {
        if let Some(ref prop) = self.properties {
            return Ok(prop);
        }
        let mut props = Default::default();
        self.base.get_properties(&mut props)?;
        Ok(self.properties.get_or_insert(Box::new(props)))
    }

    fn get_image_properties(&mut self) -> l0::Result<&l0::sys::ze_device_image_properties_t> {
        if let Some(ref prop) = self.image_properties {
            return Ok(prop);
        }
        let mut props = Default::default();
        self.base.get_image_properties(&mut props)?;
        Ok(self.image_properties.get_or_insert(Box::new(props)))
    }

    pub(crate) fn get_memory_properties(&mut self) -> l0::Result<&[l0::sys::ze_device_memory_properties_t]> {
        if let Some(ref prop) = self.memory_properties {
            return Ok(prop);
        }
        match self.base.get_memory_properties() {
            Ok(prop) => Ok(self.memory_properties.get_or_insert(prop)),
            Err(e) => Err(e),
        }
    }

    fn get_compute_properties(&mut self) -> l0::Result<&l0::sys::ze_device_compute_properties_t> {
        if let Some(ref prop) = self.compute_properties {
            return Ok(prop);
        }
        let mut props = Default::default();
        self.base.get_compute_properties(&mut props)?;
        Ok(self.compute_properties.get_or_insert(Box::new(props)))
    }

    pub fn late_init(&mut self) {
        self.primary_context.as_option_mut().unwrap().device = self as *mut _;
    }

    fn get_max_simd(&mut self) -> l0::Result<u32> {
        let props = self.get_compute_properties()?;
        Ok(*props.subGroupSizes[0..props.numSubGroupSizes as usize]
            .iter()
            .max()
            .unwrap())
    }
}

pub fn init(driver: &l0::Driver) -> Result<Vec<Device>, CUresult> {
    let ze_devices = driver.devices()?;
    let mut devices = ze_devices
        .into_iter()
        .enumerate()
        .map(|(idx, d)| unsafe { Device::new(driver, d, idx) })
        .collect::<Result<Vec<_>, _>>()?;
    for dev in devices.iter_mut() {
        dev.late_init();
        dev.primary_context.late_init();
    }
    Ok(devices)
}

pub fn get_count(count: *mut c_int) -> Result<(), CUresult> {
    let len = GlobalState::lock(|state| state.devices.len())?;
    unsafe { *count = len as c_int };
    Ok(())
}

pub fn get(device: *mut Index, ordinal: c_int) -> Result<(), CUresult> {
    if device == ptr::null_mut() || ordinal < 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let len = GlobalState::lock(|state| state.devices.len())?;
    if ordinal < (len as i32) {
        unsafe { *device = Index(ordinal) };
        Ok(())
    } else {
        Err(CUresult::CUDA_ERROR_INVALID_VALUE)
    }
}

pub fn get_name(name: *mut c_char, len: i32, dev_idx: Index) -> Result<(), CUresult> {
    if name == ptr::null_mut() || len < 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let name_ptr = GlobalState::lock_device(dev_idx, |dev| {
        let props = dev.get_properties()?;
        Ok::<_, l0::sys::ze_result_t>(props.name.as_ptr())
    })??;
    let name_len = (0..256)
        .position(|i| unsafe { *name_ptr.add(i) } == 0)
        .unwrap_or(256);
    let mut dst_null_pos = cmp::min((len - 1) as usize, name_len);
    unsafe { std::ptr::copy_nonoverlapping(name_ptr, name, dst_null_pos) };
    if name_len + PROJECT_URL_SUFFIX_LONG.len() < (len as usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                PROJECT_URL_SUFFIX_LONG.as_ptr(),
                name.add(name_len) as *mut _,
                PROJECT_URL_SUFFIX_LONG.len(),
            )
        };
        dst_null_pos += PROJECT_URL_SUFFIX_LONG.len();
    } else if name_len + PROJECT_URL_SUFFIX_SHORT.len() < (len as usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                PROJECT_URL_SUFFIX_SHORT.as_ptr(),
                name.add(name_len) as *mut _,
                PROJECT_URL_SUFFIX_SHORT.len(),
            )
        };
        dst_null_pos += PROJECT_URL_SUFFIX_SHORT.len();
    }
    unsafe { *(name.add(dst_null_pos)) = 0 };
    Ok(())
}

/// The PCI bus id of the device, in the "dddd:dd:dd.d" form that CUDA uses.
/// It comes from sysfs: the DRM node of the GPU links to its PCI device, whose
/// directory name is the bus id.
fn read_pci_bus_id(dev_idx: Index) -> Option<String> {
    let mut candidates = Vec::new();
    // The DRM card nodes: /sys/class/drm/card*/device -> ../../devices/...
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let device_link = entry.path().join("device");
            if let Ok(target) = std::fs::read_link(&device_link) {
                if let Some(bdf) = target.file_name().and_then(|f| f.to_str()) {
                    if bdf.contains(':') && bdf.contains('.') {
                        candidates.push(bdf.to_string());
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        // Fall back to any PCI device.
        if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.contains(':') && name.contains('.') {
                    candidates.push(name);
                }
            }
        }
    }
    let _ = dev_idx;
    candidates.sort();
    candidates.into_iter().next()
}

pub fn get_pci_bus_id(
    pci_bus_id: *mut c_char,
    len: c_int,
    dev_idx: Index,
) -> Result<(), CUresult> {
    if pci_bus_id.is_null() || len <= 0 {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let id = match read_pci_bus_id(dev_idx) {
        Some(id) => id,
        None => return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED),
    };
    let bytes = id.as_bytes();
    let n = cmp::min(bytes.len(), (len - 1) as usize);
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), pci_bus_id as *mut u8, n) };
    unsafe { *(pci_bus_id.add(n)) = 0 };
    Ok(())
}

pub fn total_mem_v2(bytes: *mut usize, dev_idx: Index) -> Result<(), CUresult> {
    if bytes == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    let mem_props = GlobalState::lock_device(dev_idx, |dev| {
        let mem_props = dev.get_memory_properties()?;
        Ok::<_, l0::sys::ze_result_t>(mem_props)
    })??;
    let max_mem = mem_props
        .iter()
        .map(|p| p.totalSize)
        .max()
        .ok_or(CUresult::CUDA_ERROR_ILLEGAL_STATE)?;
    unsafe { *bytes = max_mem as usize };
    Ok(())
}

pub fn get_default_mem_pool(pool: *mut cuda::CUmemoryPool, dev_idx: Index) -> Result<(), CUresult> {
    if pool == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    // Dummy opaque handle: allocations do not go through pools, and libcudart
    // only stores the handle without dereferencing it.
    let _ = dev_idx;
    static DUMMY_POOL: cuda::CUmemPoolHandle_st = unsafe { mem::zeroed() };
    unsafe { *pool = &DUMMY_POOL as *const _ as *mut cuda::CUmemPoolHandle_st };
    unsafe { *pool = &DUMMY_POOL as *const _ as *mut cuda::CUmemPoolHandle_st };
    Ok(())
}


impl CUdevice_attribute {
    fn get_static_value(self) -> Option<i32> {
        match self {
            // Discrete GPU with several hardware contexts that can overlap.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP => Some(1),
            // The desktop driver does not enforce kernel exec timeouts.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT => Some(1),
            // Claim discrete so applications pick the discrete path.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED => Some(0),
            // Answered from ZLUDA_FAKE_CC in get_attribute.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => None,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => None,
            // Host memory is pinned through L0 host allocations.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY => Some(1),
            // Constants share the global address space here.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY => Some(65536),
            // L0 reports no pitch limit.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH => Some(2147483647),
            // The register file is not exposed by Level Zero.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => Some(65536),
            // 512-byte alignment is the CUDA convention.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT => Some(512),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS => Some(1),
            // No ECC on consumer parts.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED => Some(0),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TCC_DRIVER => Some(0),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING => Some(1),
            // Priorities are accepted and ignored (see stream.rs).
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED => Some(1),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED => Some(1),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED => Some(1),

            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR => Some(65536),
            // No managed memory: no SVM on this L0 backend.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY => Some(0),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD => Some(0),
            // Host atomics on mapped memory are not implemented.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED => Some(0),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED => Some(1),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM => {
                Some(1)
            }
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS => Some(1),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS => Some(1),
            // Cooperative launch is not implemented.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH => Some(0),
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH => Some(0),

            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED => Some(1),
            // The resident-block cap the occupancy queries model.
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR => Some(16),
            _ => None,
        }
    }
}


pub fn get_attribute(
    pi: *mut i32,
    attrib: CUdevice_attribute,
    dev_idx: Index,
) -> Result<(), CUresult> {
    if pi == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    if let Some(value) = attrib.get_static_value() {
        unsafe { *pi = value };
        return Ok(());
    }
    let value: i32 = match attrib {
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT => 2,
        // A multiprocessor corresponds roughly to a sub-slice (a thread group
        // cannot cross one either).
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT => 28,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => {
            GlobalState::lock_device(dev_idx, |dev| {
                let max_simd = dev.get_max_simd()?;
                let props = dev.get_properties()?;
                Ok::<_, l0::sys::ze_result_t>(
                    (props.numEUsPerSubslice * props.numThreadsPerEU * max_simd) as i32,
                )
            })??
        }
        // The largest workgroup-local scratch the hardware has. A kernel that
        // asks for more dynamic shared memory is refused by the runtime before
        // any driver call, so the value has to be the real one.
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(props.maxSharedLocalMemory as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(cmp::min(
                    i32::max_value() as u32,
                    props.maxTotalGroupSize,
                ) as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_image_properties()?;
                Ok::<_, l0::sys::ze_result_t>(cmp::min(
                    props.maxImageDims1D,
                    c_int::max_value() as u32,
                ) as c_int)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(cmp::min(
                    i32::max_value() as u32,
                    props.maxGroupCountX,
                ) as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(cmp::min(
                    i32::max_value() as u32,
                    props.maxGroupCountY,
                ) as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(cmp::min(
                    i32::max_value() as u32,
                    props.maxGroupCountZ,
                ) as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(
                    cmp::min(i32::max_value() as u32, props.maxGroupSizeX) as i32,
                )
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(
                    cmp::min(i32::max_value() as u32, props.maxGroupSizeY) as i32,
                )
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(
                    cmp::min(i32::max_value() as u32, props.maxGroupSizeZ) as i32,
                )
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_compute_properties()?;
                Ok::<_, l0::sys::ze_result_t>(props.maxSharedLocalMemory as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => {
            GlobalState::lock_device(dev_idx, |dev| Ok::<_, CUresult>(dev.get_max_simd()? as i32))??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_properties()?;
                Ok::<_, l0::sys::ze_result_t>(props.deviceId as i32)
            })??
        }
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE => {
            GlobalState::lock_device(dev_idx, |dev| {
                let props = dev.get_properties()?;
                Ok::<_, l0::sys::ze_result_t>(props.coreClockRate as i32)
            })??
        }
        // ZLUDA_FAKE_CC=75 reports 7.5 instead of 8.0, a diagnostic: libraries
        // pick kernel configurations from the capability.
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => match std::env::var("ZLUDA_FAKE_CC").ok().as_deref() {
            Some(v) if v.len() >= 2 => v[..1].parse().unwrap_or(8),
            _ => 8,
        },
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR => match std::env::var("ZLUDA_FAKE_CC").ok().as_deref() {
            Some(v) if v.len() >= 2 => v[1..2].parse().unwrap_or(0),
            _ => 0,
        },
        // Unknown attributes read as 0, the way CUDA reports an attribute it
        // does not support; leaving them uninitialized is worse.
        _ => 0,
    };
    unsafe { *pi = value };
Ok(())
}

pub fn compute_capability(
    major: *mut i32,
    minor: *mut i32,
    _dev_idx: Index,
) -> Result<(), CUresult> {
    if major == ptr::null_mut() || minor == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    // The reported capability is the lowest common denominator for the PTX we
    // parse (sm_80), regardless of the physical hardware.
    unsafe {
        *major = 8;
        *minor = 0;
    }
    Ok(())
}

pub fn get_properties(
    prop: *mut crate::cuda::CUdevprop,
    dev_idx: Index,
) -> Result<(), CUresult> {
    if prop == ptr::null_mut() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }
    GlobalState::lock_device(dev_idx, |dev| {
let props = *dev.get_properties()?;
let compute = *dev.get_compute_properties()?;
        let max_simd = dev.get_max_simd()?;
        let max_threads =
            cmp::min(i32::max_value() as u32, compute.maxTotalGroupSize) as i32;
        let max_grid = |v: u32| cmp::min(i32::max_value() as u32, v) as i32;
        let max_block = |v: u32| cmp::min(i32::max_value() as u32, v) as i32;
        unsafe {
            (*prop).maxThreadsPerBlock = max_threads;
            (*prop).maxThreadsDim = [
                max_block(compute.maxGroupSizeX),
                max_block(compute.maxGroupSizeY),
                max_block(compute.maxGroupSizeZ),
            ];
            (*prop).maxGridSize = [
                max_grid(compute.maxGroupCountX),
                max_grid(compute.maxGroupCountY),
                max_grid(compute.maxGroupCountZ),
            ];
            (*prop).sharedMemPerBlock = compute.maxSharedLocalMemory as i32;
            (*prop).totalConstantMemory = 65536;
            (*prop).SIMDWidth = max_simd as i32;
            (*prop).memPitch = 2147483647;
            (*prop).regsPerBlock = 65536;
            (*prop).clockRate = props.coreClockRate as i32;
            (*prop).textureAlign = 512;
        }
        Ok::<_, l0::sys::ze_result_t>(())
    })??;
    Ok(())
}

pub fn get_uuid(uuid: *mut CUuuid_st, dev_idx: Index) -> Result<(), CUresult> {
    let ze_uuid = GlobalState::lock_device(dev_idx, |dev| {
        let props = dev.get_properties()?;
        Ok::<_, l0::sys::ze_result_t>(props.uuid)
    })??;
    unsafe {
        *uuid = CUuuid_st {
            bytes: mem::transmute(ze_uuid.id),
        }
    };
    Ok(())
}

// TODO: add support if Level 0 exposes it
pub fn get_luid(
    luid: *mut c_char,
    dev_node_mask: *mut c_uint,
    _dev_idx: Index,
) -> Result<(), CUresult> {
    unsafe { ptr::write_bytes(luid, 0u8, 8) };
    unsafe { *dev_node_mask = 0 };
    Ok(())
}

pub fn primary_ctx_get_state(
    dev_idx: Index,
    flags: *mut u32,
    active: *mut i32,
) -> Result<(), CUresult> {
    let (is_active, flags_value) = GlobalState::lock_device(dev_idx, |dev| {
        // This is safe because primary context can't be dropped
        let ctx_ptr = &mut dev.primary_context as *mut _;
        let flags_ptr =
            (&unsafe { dev.primary_context.as_ref_unchecked() }.flags) as *const AtomicU32;
        let is_active = context::CONTEXT_STACK
            .with(|stack| stack.borrow().last().map(|x| *x))
            .map(|current| current == ctx_ptr)
            .unwrap_or(false)
            || dev.primary_ctx_retained.load(Ordering::Relaxed);
        let flags_value = unsafe { &*flags_ptr }.load(Ordering::Relaxed);
        Ok::<_, l0::sys::ze_result_t>((is_active, flags_value))
    })??;
    unsafe { *active = if is_active { 1 } else { 0 } };
    unsafe { *flags = flags_value };
    Ok(())
}

pub fn primary_ctx_retain(
    pctx: *mut *mut context::Context,
    dev_idx: Index,
) -> Result<(), CUresult> {
    let ctx_ptr = GlobalState::lock_device(dev_idx, |dev| {
        dev.primary_ctx_retained.store(true, Ordering::Relaxed);
        &mut dev.primary_context as *mut _
    })?;
    unsafe { *pctx = ctx_ptr };
    // Retaining the primary context also makes it current for the calling
    // thread, as CUDA does, so allocs and copies can use it implicitly.
    context::set_current(ctx_ptr);
    // Also record the process-global fallback, so a thread with an empty
    // CONTEXT_STACK (a worker spawned by the runtime, a child after fork)
    // still finds a current context.
    context::set_primary_context(ctx_ptr);
    Ok(())
}

// TODO: allow for retain/reset/release of primary context
pub(crate) fn primary_ctx_release_v2(_dev_idx: Index) -> CUresult {
    CUresult::CUDA_SUCCESS
}

pub struct DynamicEventPool {
    count: usize,
    pool_flags: l0::sys::ze_event_pool_flags_t,
    signal_flags: l0::sys::ze_event_scope_flags_t,
    events: Vec<DynamicEventPoolEntry>,
}

impl DynamicEventPool {
    fn new(
        dev: l0::Device,
        ctx: &'static l0::Context,
        pool_flags: l0::sys::ze_event_pool_flags_t,
        signal_flags: l0::sys::ze_event_scope_flags_t,
    ) -> l0::Result<Self> {
        Ok(DynamicEventPool {
            count: 0,
            pool_flags,
            signal_flags,
            events: vec![DynamicEventPoolEntry::new(dev, ctx, pool_flags)?],
        })
    }

    pub fn get(
        &'static mut self,
        dev: l0::Device,
        ctx: &'static l0::Context,
    ) -> l0::Result<(l0::Event<'static>, u64)> {
        self.count += 1;
        let events = unsafe { transmute_lifetime_mut(&mut self.events) };
        let (global_idx, (ev, local_idx)) = {
            for (idx, entry) in self.events.iter_mut().enumerate() {
                if let Some((ev, local_idx)) = entry.get(self.signal_flags)? {
                    let marker = (idx << 32) as u64 | local_idx as u64;
                    return Ok((ev, marker));
                }
            }
            events.push(DynamicEventPoolEntry::new(dev, ctx, self.pool_flags)?);
            let global_idx = (events.len() - 1) as u64;
            (
                global_idx,
                events.last_mut().unwrap().get(self.signal_flags)?.unwrap(),
            )
        };
        let marker = (global_idx << 32) | local_idx as u64;
        Ok((ev, marker))
    }

    pub fn mark_as_free(&mut self, marker: u64) {
        let global_idx = (marker >> 32) as u32;
        self.events[global_idx as usize].mark_as_free(marker as u32);
        self.count -= 1;
        // TODO: clean up empty entries
    }
}

const DYNAMIC_EVENT_POOL_ENTRY_SIZE: usize = 448;
const DYNAMIC_EVENT_POOL_ENTRY_BITMAP_SIZE: usize =
    DYNAMIC_EVENT_POOL_ENTRY_SIZE / (mem::size_of::<u64>() * 8);
#[repr(C)]
#[repr(align(64))]
struct DynamicEventPoolEntry {
    event_pool: l0::EventPool<'static>,
    bit_map: [u64; DYNAMIC_EVENT_POOL_ENTRY_BITMAP_SIZE],
}

impl DynamicEventPoolEntry {
    fn new(
        dev: l0::Device,
        ctx: &'static l0::Context,
        flags: l0::sys::ze_event_pool_flags_t,
    ) -> l0::Result<Self> {
        Ok(DynamicEventPoolEntry {
            event_pool: l0::EventPool::new(
                ctx,
                flags,
                DYNAMIC_EVENT_POOL_ENTRY_SIZE as u32,
                Some(&[dev]),
            )?,
            bit_map: [0; DYNAMIC_EVENT_POOL_ENTRY_BITMAP_SIZE],
        })
    }

    fn get(
        &'static mut self,
        signal: l0::sys::ze_event_scope_flags_t,
    ) -> l0::Result<Option<(l0::Event<'static>, u32)>> {
        for (idx, value) in self.bit_map.iter_mut().enumerate() {
            let shift = first_index_of_zero_u64(*value);
            if shift == 64 {
                continue;
            }
            *value = *value | (1u64 << shift);
            let entry_index = (idx as u32 * 64u32) + shift;
            let event = l0::Event::new(
                &self.event_pool,
                entry_index,
                signal,
                l0::sys::ze_event_scope_flags_t(0),
            )?;
            return Ok(Some((event, entry_index)));
        }
        Ok(None)
    }

    fn mark_as_free(&mut self, idx: u32) {
        let value = &mut self.bit_map[idx as usize / 64];
        let shift = idx % 64;
        *value = *value & !(1 << shift);
    }
}

fn first_index_of_zero_u64(x: u64) -> u32 {
    let x = !x;
    (x & x.wrapping_neg()).trailing_zeros()
}

#[cfg(test)]
mod test {
    use std::mem;

    use super::DynamicEventPoolEntry;

    use super::super::test::CudaDriverFns;
    use super::super::CUresult;

    cuda_driver_test!(primary_ctx_default_inactive);

    fn primary_ctx_default_inactive<T: CudaDriverFns>() {
        assert_eq!(T::cuInit(0), CUresult::CUDA_SUCCESS);
        let mut flags = u32::max_value();
        let mut active = i32::max_value();
        assert_eq!(
            T::cuDevicePrimaryCtxGetState(0, &mut flags, &mut active),
            CUresult::CUDA_SUCCESS
        );
        assert_eq!(flags, 0);
        assert_eq!(active, 0);
    }

    #[test]
    pub fn dynamic_event_pool_page_is_64b() {
        assert_eq!(mem::size_of::<DynamicEventPoolEntry>(), 64);
        assert_eq!(mem::align_of::<DynamicEventPoolEntry>(), 64);
    }
}

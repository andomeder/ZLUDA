use crate::cuda::CUresult;
use crate::r#impl::os;
use crate::{
    cuda::{CUcontext, CUdevice, CUmodule, CUuuid},
    cuda_impl,
};

use super::{context, context::ContextData, device, module, Decuda, Encuda, GlobalState};
use std::os::raw::{c_uint, c_ulong, c_ushort};
use std::{
    ffi::{c_void, CStr},
    ptr,
};
use std::{mem, os::raw::c_int};

pub fn get(table: *mut *const std::os::raw::c_void, id: *const CUuuid) -> CUresult {
    if table == ptr::null_mut() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    // Some cudart versions call cuGetExportTable with a null GUID and expect
    // the CUDART_INTERFACE table rather than an error.
    if id == ptr::null_mut() {
        unsafe { *table = CUDART_INTERFACE_VTABLE.as_ptr() as *const _ };
        return CUresult::CUDA_SUCCESS;
    }
    let id = unsafe { *id };
    match id {
        TOOLS_RUNTIME_CALLBACK_HOOKS_GUID => {
            unsafe { *table = TOOLS_RUNTIME_CALLBACK_HOOKS_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        CUDART_INTERFACE_GUID => {
            unsafe { *table = CUDART_INTERFACE_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        TOOLS_TLS_GUID => {
            unsafe { *table = TOOLS_TLS_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        CONTEXT_LOCAL_STORAGE_INTERFACE_V0301_GUID => {
            unsafe { *table = CONTEXT_LOCAL_STORAGE_INTERFACE_V0301_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        CTX_CREATE_BYPASS_GUID => {
            unsafe { *table = CTX_CREATE_BYPASS_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        HEAP_ACCESS_GUID => {
            unsafe { *table = HEAP_ACCESS_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        INTEGRITY_CHECK_GUID => {
            unsafe { *table = INTEGRITY_CHECK_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        CONTEXT_CHECKS_GUID => {
            unsafe { *table = CONTEXT_CHECKS_VTABLE.as_ptr() as *const _ };
            CUresult::CUDA_SUCCESS
        }
        _ => {
            CUresult::CUDA_ERROR_NOT_SUPPORTED
        }
    }
}

const TOOLS_RUNTIME_CALLBACK_HOOKS_GUID: CUuuid = CUuuid {
    bytes: [
        0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74, 0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a,
        0x66,
    ],
};
#[repr(C)]
union VTableEntry {
    ptr: *const (),
    length: usize,
}
unsafe impl Sync for VTableEntry {}
const TOOLS_RUNTIME_CALLBACK_HOOKS_LENGTH: usize = 7;
static TOOLS_RUNTIME_CALLBACK_HOOKS_VTABLE: [VTableEntry; TOOLS_RUNTIME_CALLBACK_HOOKS_LENGTH] = [
    VTableEntry {
        length: mem::size_of::<[VTableEntry; TOOLS_RUNTIME_CALLBACK_HOOKS_LENGTH]>(),
    },
    VTableEntry {
        ptr: runtime_callback_hooks_fn2 as *const (),
    },
    VTableEntry {
        ptr: runtime_callback_hooks_fn1 as *const (),
    },
    VTableEntry {
        // A no-op: cudart calls this slot with three pointers during a launch
        // and ignores the result.
        ptr: runtime_callback_hooks_fn3 as *const (),
    },
    VTableEntry {
        ptr: runtime_callback_hooks_fn4 as *const (),
    },
    VTableEntry { ptr: ptr::null() },
    VTableEntry {
        ptr: runtime_callback_hooks_fn5 as *const (),
    },
];

// Scratch space handed to cudart through the "unknown buffer" hooks: cudart
// treats it as an opaque driver-owned block and reads a capability word out of
// it. cudart 12.4 reads a u32 at 0x34c and refuses to launch
// (cudaErrorNotSupported) while it is zero, so a non-zero value is published
// there.
const TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_CAPABILITY_OFFSET: usize = 0x34c;
const TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_CAPABILITY_VALUE: u32 = 1;
static mut TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_SPACE: [usize; 512] = [0; 512];

unsafe extern "system" fn runtime_callback_hooks_fn3(
    _a: *mut std::ffi::c_void,
    _b: *mut std::ffi::c_void,
    _c: *mut std::ffi::c_void,
) -> u32 {
    0
}

unsafe extern "system" fn runtime_callback_hooks_fn1(ptr: *mut *mut usize, size: *mut usize) {
    let space = TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_SPACE.as_mut_ptr();
    // Publish the capability word exactly once, on the first handout.
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        std::ptr::write_unaligned(
            (space as *mut u8).add(TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_CAPABILITY_OFFSET)
                as *mut u32,
            TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_CAPABILITY_VALUE,
        );
    });
    *ptr = space;
    *size = TOOLS_RUNTIME_CALLBACK_HOOKS_FN1_SPACE.len();
}

// Publish a per-thread status for the runtime's callback hooks. Every
// operation cudart asks about here is one we support, so the status is 0.
unsafe extern "system" fn runtime_callback_hooks_fn2(
    _code: c_int,
    status: *mut c_int,
    _count: usize,
) -> c_int {
    if !status.is_null() {
        *status = 0;
    }
    0
}

// Same shape as fn2 and used while cudart builds a launch config. The status
// slot must be written: leaving it untouched makes cudart read a stale value
// and abort the launch.
unsafe extern "system" fn runtime_callback_hooks_fn4(
    _code: c_int,
    status: *mut c_int,
    _size: usize,
) -> c_int {
    if !status.is_null() {
        *status = 0;
    }
    0
}

static mut TOOLS_RUNTIME_CALLBACK_HOOKS_FN5_SPACE: [u8; 16] = [0; 16];

// The size reported here has to be > 13 (0xd), otherwise cudart treats the
// tools-callback-hooks interface as invalid and fails init with
// cudaErrorInsufficientDriver.
unsafe extern "system" fn runtime_callback_hooks_fn5(
    ptr: *mut *mut u8,
    size: *mut usize,
) -> *mut u8 {
    *ptr = TOOLS_RUNTIME_CALLBACK_HOOKS_FN5_SPACE.as_mut_ptr();
    *size = TOOLS_RUNTIME_CALLBACK_HOOKS_FN5_SPACE.len();
    return TOOLS_RUNTIME_CALLBACK_HOOKS_FN5_SPACE.as_mut_ptr();
}

const CONTEXT_CHECKS_GUID: CUuuid = CUuuid {
    bytes: [
        0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43, 0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa,
        0x7e,
    ],
};

const CONTEXT_CHECKS_LENGTH: usize = 4;
static CONTEXT_CHECKS_VTABLE: [VTableEntry; CONTEXT_CHECKS_LENGTH] = [
    VTableEntry {
        length: mem::size_of::<[VTableEntry; CONTEXT_CHECKS_LENGTH]>(),
    },
    VTableEntry { ptr: ptr::null() },
    VTableEntry {
        ptr: context_check as *const (),
    },
    VTableEntry {
        ptr: check_fn3 as *const (),
    },
];

// cudart calls both of these during init; both answer the same way upstream
// ZLUDA's dark_api does (zero and one).
unsafe extern "system" fn context_check(
    _ctx_in: CUcontext,
    result1: *mut u32,
    result2: *mut *const std::ffi::c_void,
) -> CUresult {
    if !result1.is_null() {
        *result1 = 0;
    }
    if !result2.is_null() {
        *result2 = ptr::null();
    }
    CUresult::CUDA_SUCCESS
}

unsafe extern "system" fn check_fn3() -> u32 {
    1
}

const CUDART_INTERFACE_GUID: CUuuid = CUuuid {
    bytes: [
        0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a, 0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d,
        0xf9,
    ],
};

const CUDART_INTERFACE_LENGTH: usize = 13;
static CUDART_INTERFACE_VTABLE: [VTableEntry; CUDART_INTERFACE_LENGTH] = [
    VTableEntry {
        length: mem::size_of::<[VTableEntry; CUDART_INTERFACE_LENGTH]>(),
    },
    VTableEntry {
        ptr: get_module_from_cubin as *const (),
    },
    VTableEntry {
        ptr: cudart_interface_fn1 as *const (),
    },
    VTableEntry { ptr: ptr::null() },
    VTableEntry { ptr: ptr::null() },
    VTableEntry { ptr: ptr::null() },
    VTableEntry {
        ptr: get_module_from_cubin_ext as *const (),
    },
    VTableEntry {
        ptr: cudart_interface_fn6 as *const (),
    },
    VTableEntry { ptr: ptr::null() },
    VTableEntry { ptr: ptr::null() },
    VTableEntry { ptr: ptr::null() },
    VTableEntry { ptr: ptr::null() },
    VTableEntry {
        ptr: load_compilers as *const (),
    },
];

unsafe extern "system" fn load_compilers() -> CUresult {
    CUresult::CUDA_SUCCESS
}

unsafe extern "system" fn cudart_interface_fn1(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    cudart_interface_fn1_impl(pctx.decuda(), dev.decuda()).encuda()
}

fn cudart_interface_fn1_impl(
    pctx: *mut *mut context::Context,
    dev: device::Index,
) -> Result<(), CUresult> {
    let ctx_ptr = GlobalState::lock_device(dev, |d| &mut d.primary_context as *mut _)?;
    unsafe { *pctx = ctx_ptr };
    Ok(())
}

const FATBINC_MAGIC: c_uint = 0x466243B1;
const FATBINC_VERSION: c_uint = 0x1;

#[repr(C)]
struct FatbincWrapper {
    magic: c_uint,
    version: c_uint,
    data: *const FatbinHeader,
    filename_or_fatbins: *const c_void,
}

const FATBIN_MAGIC: c_uint = 0xBA55ED50;
const FATBIN_VERSION: c_ushort = 0x01;

#[repr(C, align(8))]
struct FatbinHeader {
    magic: c_uint,
    version: c_ushort,
    header_size: c_ushort,
    files_size: c_ulong, // excluding frame header, size of all blocks framed by this frame
}

unsafe extern "system" fn get_module_from_cubin(
    result: *mut CUmodule,
    fatbinc_wrapper: *const FatbincWrapper,
) -> CUresult {
    if result == ptr::null_mut() || fatbinc_wrapper.is_null() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if (*fatbinc_wrapper).magic != FATBINC_MAGIC
        || (*fatbinc_wrapper).version != FATBINC_VERSION
    {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    // The parsing itself lives in impl::fatbin: one source of truth for the
    // variable-size record headers and the LZ4-compressed PTX records.
    module::load_fatbin(result.decuda(), fatbinc_wrapper as *const c_void).encuda()
}

// Cudart 12.4 calls this with the fatbinc wrapper plus three extra
// out-parameters that are always non-null; they are outputs only, so the
// module is loaded regardless and the caller overwrites them.
unsafe extern "system" fn get_module_from_cubin_ext(
    result: *mut CUmodule,
    fatbinc_wrapper: *const FatbincWrapper,
    _arg3: *mut c_void,
    _arg4: *mut c_void,
) -> CUresult {
    get_module_from_cubin(result, fatbinc_wrapper)
}

unsafe extern "system" fn cudart_interface_fn6(_: u64) {}

const TOOLS_TLS_GUID: CUuuid = CUuuid {
    bytes: [
        0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec,
        0xdc,
    ],
};

const TOOLS_TLS_LENGTH: usize = 5;
static TOOLS_TLS_VTABLE: [VTableEntry; TOOLS_TLS_LENGTH] = [
    VTableEntry { length: mem::size_of::<[VTableEntry; TOOLS_TLS_LENGTH]>() },
    VTableEntry { ptr: tools_tls_get as *const () },
    VTableEntry { ptr: tools_tls_set as *const () },
    VTableEntry { ptr: tools_tls_destroy as *const () },
    VTableEntry { ptr: tools_tls_get as *const () },
];

// Per-thread slots used by cudart's runtime to stash context-local state while
// it assembles a launch configuration. The keys are opaque pointers owned by
// cudart.
//
//   [1] get(key)          -> stored value, or NULL
//   [2] set(key, value)   -> stores the value for the calling thread
//   [3] destroy(value)    -> releases a value handed out by get()
//   [4] get(key)          -> same accessor, second key domain
thread_local! {
    static TOOLS_TLS_VALUES: std::cell::RefCell<std::collections::HashMap<usize, usize>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

unsafe extern "system" fn tools_tls_get(key: usize) -> *mut c_void {
    TOOLS_TLS_VALUES.with(|values| {
        values
            .borrow()
            .get(&key)
            .map(|value| *value as *mut c_void)
            .unwrap_or(ptr::null_mut())
    })
}

unsafe extern "system" fn tools_tls_set(key: usize, value: *mut c_void) {
    TOOLS_TLS_VALUES.with(|values| {
        values.borrow_mut().insert(key, value as usize);
    });
}

unsafe extern "system" fn tools_tls_destroy(value: *mut c_void) {
    TOOLS_TLS_VALUES.with(|values| {
        values.borrow_mut().retain(|_, stored| *stored != value as usize);
    });
}

const CONTEXT_LOCAL_STORAGE_INTERFACE_V0301_GUID: CUuuid = CUuuid {
    bytes: [
        0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11, 0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95,
        0x93,
    ],
};

// cudart 12.x uses the modern keyed interface (upstream ZLUDA dark_api):
//   [0] put(context, key, value, dtor_cb)
//   [1] delete(context, key)
//   [2] get(value_out, context, key)
static CONTEXT_LOCAL_STORAGE_INTERFACE_V0301_VTABLE: [VTableEntry; 4] = [
    VTableEntry {
        ptr: context_local_storage_put as *const (),
    },
    VTableEntry {
        ptr: context_local_storage_delete as *const (),
    },
    VTableEntry {
        ptr: context_local_storage_get as *const (),
    },
    VTableEntry { ptr: ptr::null() },
];

unsafe extern "system" fn context_local_storage_put(
    cu_ctx: CUcontext,
    key: *mut std::ffi::c_void,
    value: *mut std::ffi::c_void,
    dtor_cb: Option<
        extern "system" fn(CUcontext, *mut std::ffi::c_void, *mut std::ffi::c_void),
    >,
) -> CUresult {
    context_local_storage_put_impl(cu_ctx.decuda(), key as usize, value as usize, dtor_cb).encuda()
}

fn context_local_storage_put_impl(
    cu_ctx: *mut context::Context,
    key: usize,
    value: usize,
    dtor_cb: Option<extern "system" fn(CUcontext, *mut std::ffi::c_void, *mut std::ffi::c_void)>,
) -> Result<(), CUresult> {
    if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
        eprintln!("local_storage put key={:#x} value={:#x}", key, value);
    }
    let _ = cu_ctx;
    local_storage()
        .lock()
        .unwrap()
        .insert(key, context::LocalStorageEntry { value, reset_cb: dtor_cb });
    Ok(())
}

unsafe extern "system" fn context_local_storage_delete(
    cu_ctx: CUcontext,
    key: *mut std::ffi::c_void,
) -> CUresult {
    context_local_storage_delete_impl(cu_ctx.decuda(), key as usize).encuda()
}

fn context_local_storage_delete_impl(
    cu_ctx: *mut context::Context,
    key: usize,
) -> Result<(), CUresult> {
    let _ = cu_ctx;
    local_storage().lock().unwrap().remove(&key);
    Ok(())
}

unsafe extern "system" fn context_local_storage_get(
    value: *mut *mut std::ffi::c_void,
    cu_ctx: CUcontext,
    key: *mut std::ffi::c_void,
) -> CUresult {
    context_local_storage_get_impl(value, cu_ctx.decuda(), key as usize).encuda()
}

fn context_local_storage_get_impl(
    value: *mut *mut std::ffi::c_void,
    cu_ctx: *mut context::Context,
    key: usize,
) -> Result<(), CUresult> {
    if std::env::var_os("ZLUDA_TRACE_LIBRARY").is_some() {
        eprintln!("local_storage get key={:#x}", key);
    }
    let _ = cu_ctx;
    match local_storage().lock().unwrap().get(&key) {
        Some(entry) => {
            unsafe { *value = entry.value as *mut std::ffi::c_void };
            Ok(())
        }
        None => Err(CUresult::CUDA_ERROR_INVALID_HANDLE),
    }
}

// The runtime stores its per-context state here keyed by the address of its
// own variable, and does so while registering fatbins - before any CUDA
// context exists. The storage therefore cannot live in the context.
fn local_storage() -> &'static std::sync::Mutex<std::collections::HashMap<usize, context::LocalStorageEntry>> {
    static STORAGE: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<usize, context::LocalStorageEntry>>,
    > = std::sync::LazyLock::new(|| {
        std::sync::Mutex::new(std::collections::HashMap::new())
    });
    &STORAGE
}

fn lock_context<T>(
    cu_ctx: *mut context::Context,
    fn_impl: impl FnOnce(&mut ContextData) -> T,
) -> Result<T, CUresult> {
    if cu_ctx == ptr::null_mut() {
        GlobalState::lock_current_context(fn_impl)
    } else {
        GlobalState::lock(|_| {
            let ctx = unsafe { &mut *cu_ctx }.as_result_mut()?;
            Ok(fn_impl(ctx))
        })?
    }
}


const CTX_CREATE_BYPASS_GUID: CUuuid = CUuuid {
    bytes: [
        0x0C, 0xA5, 0x0B, 0x8C, 0x10, 0x04, 0x92, 0x9A, 0x89, 0xA7, 0xD0, 0xDF, 0x10, 0xE7, 0x72,
        0x86,
    ],
};

const CTX_CREATE_BYPASS_LENGTH: usize = 2;
static CTX_CREATE_BYPASS_VTABLE: [VTableEntry; CTX_CREATE_BYPASS_LENGTH] = [
    VTableEntry {
        length: mem::size_of::<[VTableEntry; CTX_CREATE_BYPASS_LENGTH]>(),
    },
    VTableEntry {
        ptr: ctx_create_v2_bypass as *const (),
    },
];

// PhysX uses this and cuCtxCreate_v2 interchangeably, so it forwards.
extern "system" fn ctx_create_v2_bypass(
    pctx: *mut CUcontext,
    flags: ::std::os::raw::c_uint,
    dev: CUdevice,
) -> CUresult {
    context::create_v2(pctx.decuda(), flags, dev.decuda()).encuda()
}

const HEAP_ACCESS_GUID: CUuuid = CUuuid {
    bytes: [
        0x19, 0x5B, 0xCB, 0xF4, 0xD6, 0x7D, 0x02, 0x4A, 0xAC, 0xC5, 0x1D, 0x29, 0xCE, 0xA6, 0x31,
        0xAE,
    ],
};

#[repr(C)]
struct HeapAllocRecord {
    arg1: usize,
    arg2: usize,
    _unknown: usize,
    global_heap: *mut c_void,
}

const HEAP_ACCESS_LENGTH: usize = 3;
static HEAP_ACCESS_VTABLE: [VTableEntry; HEAP_ACCESS_LENGTH] = [
    VTableEntry {
        length: mem::size_of::<[VTableEntry; HEAP_ACCESS_LENGTH]>(),
    },
    VTableEntry {
        ptr: heap_alloc as *const (),
    },
    VTableEntry {
        ptr: heap_free as *const (),
    },
];

// Not modelled: the heap is a null allocator (see os_unix), so these only
// keep the table's shape for callers that fetch it.
unsafe extern "system" fn heap_alloc(
    halloc_ptr: *mut *const HeapAllocRecord,
    arg1: usize,
    arg2: usize,
) -> CUresult {
    if halloc_ptr == ptr::null_mut() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let halloc = GlobalState::lock(|global_state| {
        let halloc = os::heap_alloc(global_state.global_heap, mem::size_of::<HeapAllocRecord>())
            as *mut HeapAllocRecord;
        if halloc == ptr::null_mut() {
            return Err(CUresult::CUDA_ERROR_OUT_OF_MEMORY);
        }
        (*halloc).arg1 = arg1;
        (*halloc).arg2 = arg2;
        (*halloc)._unknown = 0;
        (*halloc).global_heap = global_state.global_heap;
        Ok(halloc)
    });
    match halloc {
        Ok(Ok(halloc)) => {
            *halloc_ptr = halloc;
            CUresult::CUDA_SUCCESS
        }
        Err(err) | Ok(Err(err)) => err,
    }
}

// See heap_alloc.
unsafe extern "system" fn heap_free(halloc: *mut HeapAllocRecord, arg1: *mut usize) -> CUresult {
    if halloc == ptr::null_mut() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    if arg1 != ptr::null_mut() {
        *arg1 = (*halloc).arg2;
    }
    GlobalState::lock(|global_state| {
        os::heap_free(global_state.global_heap, halloc as *mut _);
        ()
    })
    .encuda()
}

const INTEGRITY_CHECK_GUID: CUuuid = CUuuid {
    bytes: [
        0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b, 0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1,
        0xf2,
    ],
};

const INTEGRITY_CHECK_LENGTH: usize = 3;
static INTEGRITY_CHECK_VTABLE: [VTableEntry; INTEGRITY_CHECK_LENGTH] = [
    VTableEntry {
        length: mem::size_of::<[VTableEntry; INTEGRITY_CHECK_LENGTH]>(),
    },
    VTableEntry {
        ptr: integrity_check as *const (),
    },
    VTableEntry { ptr: ptr::null() },
];

// CUDA driver integrity check (NVIDIA's anti-tamper handshake): libcudart
// requests this table during init and validates the returned hash. Ported from
// upstream ZLUDA's dark_api.
unsafe extern "system" fn integrity_check(
    version: u32,
    unix_seconds: u64,
    result: *mut [u64; 2],
) -> CUresult {
    let current_process = std::process::id();
    let current_process = std::process::id();
    let current_thread = os::current_thread();

    let integrity_check_table = INTEGRITY_CHECK_VTABLE.as_ptr().cast();
    let cudart_table = CUDART_INTERFACE_VTABLE.as_ptr().cast();
    let fn_address = INTEGRITY_CHECK_VTABLE[1].ptr;

    let devices = match get_device_hash_info() {
        Ok(devices) => devices,
        Err(err) => return err,
    };
    let device_count = devices.len() as u32;
    let get_device = |dev| devices[dev as usize];

    let hash = integrity_check_impl(
        version,
        unix_seconds,
        crate::r#impl::driver_get_version() as u32,
        current_process,
        current_thread,
        integrity_check_table,
        cudart_table,
        fn_address as *const c_void,
        device_count,
        get_device,
    );
    unsafe { *result = hash };
    CUresult::CUDA_SUCCESS
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceHashinfo {
    guid: CUuuid,
    pci_domain: i32,
    pci_bus: i32,
    pci_device: i32,
}

fn get_device_hash_info() -> Result<Vec<DeviceHashinfo>, CUresult> {
    let mut device_count = 0;
    device::get_count(&mut device_count)?;

    (0..device_count)
        .map(|dev| {
            let mut guid = unsafe { mem::zeroed() };
            device::get_uuid(&mut guid, device::Index(dev))?;

            let mut pci_domain = 0;
            device::get_attribute(
                &mut pci_domain,
                crate::cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                device::Index(dev),
            )?;

            let mut pci_bus = 0;
            device::get_attribute(
                &mut pci_bus,
                crate::cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                device::Index(dev),
            )?;

            let mut pci_device = 0;
            device::get_attribute(
                &mut pci_device,
                crate::cuda::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                device::Index(dev),
            )?;

            Ok(DeviceHashinfo {
                guid,
                pci_domain,
                pci_bus,
                pci_device,
            })
        })
        .collect()
}

fn integrity_check_impl(
    version: u32,
    unix_seconds: u64,
    driver_version: u32,
    current_process: u32,
    current_thread: u32,
    integrity_check_table: *const c_void,
    cudart_table: *const c_void,
    fn_address: *const c_void,
    devices: u32,
    get_device: impl FnMut(u32) -> DeviceHashinfo,
) -> [u64; 2] {
    match version % 10 {
        0 => return [0x3341181C03CB675C, 0x8ED383AA1F4CD1E8],
        1 => return [0x1841181C03CB675C, 0x8ED383AA1F4CD1E8],
        _ => {}
    }
    // The first pass uses no input, so its result is a constant.
    let pass1_result = [
        0x14u8, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52, 0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A,
        0x9F,
    ];
    let mut result = [0u8; 66];
    pass2(&mut result, &pass1_result);
    let pass3_input = Pass3Input {
        driver_version,
        version,
        current_process,
        current_thread,
        cudart_table,
        integrity_check_table,
        fn_address,
        unix_seconds,
    };
    pass3(&mut result, &pass3_input);
    pass4(&mut result, devices, get_device);
    let pass5_1 = pass5(&mut result);
    zero_result(&mut result);
    pass6(&mut result, &pass1_result);
    pass7(&mut result, &pass5_1);
    pass5(&mut result)
}

fn pass7(accumulator: &mut [u8; 66], pass5_1: &[u64; 2]) {
    hash_pass(accumulator, pass5_1, 0);
}

fn pass6(accumulator: &mut [u8; 66], pass1_result: &[u8; 16]) {
    hash_pass(accumulator, pass1_result, 0x5c);
}

fn zero_result(result: &mut [u8; 66]) {
    for i in 0..16 {
        result[i] = 0;
    }
    for i in 48..66 {
        result[i] = 0;
    }
}

fn pass5(result: &mut [u8; 66]) -> [u64; 2] {
    let temp = 16u8.wrapping_sub(result[64]);
    for _ in 0..temp {
        integrity_check_single_pass(result, temp);
    }
    let mut temp_ptr = unsafe { result.as_mut_ptr().add(0x30) };
    loop {
        let temp = unsafe { *temp_ptr };
        temp_ptr = unsafe { temp_ptr.add(1) };
        integrity_check_single_pass(result, temp);
        if temp_ptr == unsafe { result.as_mut_ptr().add(0x40) } {
            break;
        }
    }
    let mut out0 = [0u8; 8];
    out0.copy_from_slice(&result[0..8]);
    let mut out1 = [0u8; 8];
    out1.copy_from_slice(&result[8..16]);
    [
        u64::from_ne_bytes(out0),
        u64::from_ne_bytes(out1),
    ]
}

#[repr(C)]
struct Pass3Input {
    driver_version: u32,
    version: u32,
    current_process: u32,
    current_thread: u32,
    cudart_table: *const c_void,
    integrity_check_table: *const c_void,
    fn_address: *const c_void,
    unix_seconds: u64,
}

fn pass2(accumulator: &mut [u8; 66], pass1_result: &[u8; 16]) {
    hash_pass(accumulator, pass1_result, 0x36)
}

fn pass3(accumulator: &mut [u8; 66], mixin: &Pass3Input) {
    hash_pass(accumulator, mixin, 0)
}

fn pass4(
    accumulator: &mut [u8; 66],
    devices: u32,
    mut get_device: impl FnMut(u32) -> DeviceHashinfo,
) {
    for dev in 0..devices {
        hash_pass(accumulator, &(get_device)(dev), 0)
    }
}

fn hash_pass<T: Sized>(accumulator: &mut [u8; 66], mixin: &T, xor_mask: u8) {
    for i in 0..std::mem::size_of_val(mixin) {
        integrity_check_single_pass(
            accumulator,
            unsafe { *std::ptr::from_ref(mixin).cast::<u8>().add(i) } ^ xor_mask,
        );
    }
}

fn integrity_check_single_pass(arg1: &mut [u8; 66], arg2: u8) {
    const MIXING_TABLE: [u8; 256] = [
        0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1, 0xEC, 0xF0, 0x06,
        0x13, 0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C, 0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C,
        0x82, 0xCA, 0x1E, 0x9B, 0x57, 0x3C, 0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A,
        0x17, 0xE5, 0x12, 0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E,
        0xBB, 0x2F, 0xEE, 0x7A, 0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F, 0x94, 0xC2, 0x10,
        0x89, 0x0B, 0x22, 0x5F, 0x21, 0x80, 0x7F, 0x5D, 0x9A, 0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E,
        0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03, 0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7,
        0x5E, 0x92, 0x2A, 0xAC, 0x56, 0xAA, 0xC6, 0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6,
        0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1, 0x45, 0x9D, 0x70, 0x59, 0x64, 0x71, 0x87,
        0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02, 0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0,
        0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7E, 0x0F, 0x55, 0x47, 0xA3, 0x23, 0xDD,
        0x51, 0xAF, 0x3A, 0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26, 0x2C, 0x53, 0x0D, 0x6E,
        0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52, 0x6A, 0xDC, 0x37,
        0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08, 0x0C, 0xBD, 0xB1, 0x4A, 0x78, 0x88,
        0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D, 0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39, 0xF2,
        0xEF, 0xB7, 0x0E, 0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
        0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33, 0x9F, 0x11, 0x83,
        0x14,
    ];
    let temp1 = arg1[0x40];
    arg1[temp1 as usize + 0x10] = arg2;
    let temp2 = temp1 as usize;
    let temp3 = (temp1 + 1) & 0xf;
    arg1[temp1 as usize + 0x20] = arg1[temp2] ^ arg2;
    let temp4 = MIXING_TABLE[(arg2 ^ arg1[0x41]) as usize];
    let temp1 = arg1[temp2 + 0x30];
    arg1[temp2 + 0x30] = temp4 ^ temp1;
    arg1[0x41] = temp4 ^ temp1;
    arg1[0x40] = temp3;
    if temp3 != 0 {
        return;
    }
    let mut temp1 = 0x29;
    let mut temp5 = 0x0;
    unsafe {
        loop {
            temp1 = temp1 ^ arg1[0];
            arg1[0] = temp1;
            let mut temp6 = arg1.as_mut_ptr().add(1);
            loop {
                let temp7 = temp6.add(1);
                temp1 = *temp6 ^ MIXING_TABLE[temp1 as usize];
                *temp6 = temp1;
                temp6 = temp7;
                if temp7 == arg1.as_mut_ptr().add(0x30) {
                    break;
                }
            }
            temp1 = temp1.wrapping_add(temp5);
            temp5 = temp5.wrapping_add(0x01);
            if temp5 == 0x12 {
                break;
            }
            temp1 = MIXING_TABLE[temp1 as usize];
        }
    }
}

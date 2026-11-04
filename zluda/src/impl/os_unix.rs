use std::ffi::c_void;
use std::ptr;

pub unsafe fn heap_create() -> *mut c_void {
    usize::MAX as *mut _
}

pub unsafe fn heap_alloc(_heap: *mut c_void, _bytes: usize) -> *mut c_void {
    // The HEAP_ACCESS dark API is not implemented (upstream ZLUDA also returns
    // an error here). Returning null makes callers fail cleanly instead of
    // panicking through the FFI boundary.
    ptr::null_mut()
}

pub unsafe fn heap_free(_heap: *mut c_void, _alloc: *mut c_void) {}

// CUDA's integrity check mixes the current thread id into its hash. Upstream
// ZLUDA computes this from pthread_self().
#[link(name = "pthread")]
unsafe extern "C" {
    fn pthread_self() -> std::os::unix::thread::RawPthread;
}

pub(crate) fn current_thread() -> u32 {
    (unsafe { pthread_self() }) as u32
}

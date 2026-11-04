// pointer_attr_test.c: verify cuPointerGetAttribute/cuPointerGetAttributes
// classifies device and host-pinned allocations correctly.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                             \
    do {                                                                        \
        CUresult err = (call);                                                  \
        if (err != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, err);                                                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr dptr = 0;
    void *hptr = NULL;
    size_t size = 4096;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));

    // --- device pointer ---
    CHECK(cuMemAlloc(&dptr, size));

    CUdeviceptr dbase = 0;
    CHECK(cuPointerGetAttribute(&dbase, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, dptr));
    if (dbase != dptr) {
        fprintf(stderr, "FAIL: RANGE_START_ADDR base mismatch %p != %p\n",
                (void *)dbase, (void *)dptr);
        exit(1);
    }
    size_t dsize = 0;
    CHECK(cuPointerGetAttribute(&dsize, CU_POINTER_ATTRIBUTE_RANGE_SIZE, dptr));
    if (dsize != size) {
        fprintf(stderr, "FAIL: RANGE_SIZE %zu != %zu\n", dsize, size);
        exit(1);
    }
    int mtype = 0;
    CHECK(cuPointerGetAttribute(&mtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr));
    if (mtype != CU_MEMORYTYPE_DEVICE) {
        fprintf(stderr, "FAIL: device ptr MEMORY_TYPE %d != CU_MEMORYTYPE_DEVICE(%d)\n",
                mtype, CU_MEMORYTYPE_DEVICE);
        exit(1);
    }
    int ordinal = -1;
    CHECK(cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dptr));
    if (ordinal != 0) {
        fprintf(stderr, "FAIL: DEVICE_ORDINAL %d != 0\n", ordinal);
        exit(1);
    }
    void *host_ptr = (void *)0x1;
    CHECK(cuPointerGetAttribute(&host_ptr, CU_POINTER_ATTRIBUTE_HOST_POINTER, dptr));
    if (host_ptr != NULL) {
        fprintf(stderr, "FAIL: device ptr HOST_POINTER should be NULL\n");
        exit(1);
    }

    // Interior pointer lookup
    CUdeviceptr interior = dptr + size / 2;
    CHECK(cuPointerGetAttribute(&dbase, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, interior));
    if (dbase != dptr) {
        fprintf(stderr, "FAIL: interior RANGE_START_ADDR mismatch\n");
        exit(1);
    }

    // --- host pinned pointer ---
    CHECK(cuMemAllocHost(&hptr, size));
    CUdeviceptr hdptr = 0;
    CHECK(cuPointerGetAttribute(&hdptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)hptr));
    if (hdptr != (CUdeviceptr)hptr) {
        fprintf(stderr, "FAIL: host DEVICE_POINTER mismatch\n");
        exit(1);
    }
    CHECK(cuPointerGetAttribute(&mtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)hptr));
    if (mtype != CU_MEMORYTYPE_HOST) {
        fprintf(stderr, "FAIL: host ptr MEMORY_TYPE %d != CU_MEMORYTYPE_HOST(%d)\n",
                mtype, CU_MEMORYTYPE_HOST);
        exit(1);
    }
    void *hh = NULL;
    CHECK(cuPointerGetAttribute(&hh, CU_POINTER_ATTRIBUTE_HOST_POINTER, (CUdeviceptr)hptr));
    if (hh != hptr) {
        fprintf(stderr, "FAIL: host HOST_POINTER mismatch\n");
        exit(1);
    }
    CHECK(cuPointerGetAttribute(&dbase, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)hptr));
    if (dbase != (CUdeviceptr)hptr) {
        fprintf(stderr, "FAIL: host RANGE_START_ADDR mismatch\n");
        exit(1);
    }

    // --- cuPointerGetAttributes (multi) ---
    CUpointer_attribute attrs[3] = {
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        CU_POINTER_ATTRIBUTE_RANGE_SIZE,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    };
    CUdeviceptr a_base = 0;
    size_t a_size = 0;
    int a_mtype = 0;
    void *adata[3] = {&a_base, &a_size, &a_mtype};
    CHECK(cuPointerGetAttributes(3, attrs, adata, dptr));
    if (a_base != dptr || a_size != size || a_mtype != CU_MEMORYTYPE_DEVICE) {
        fprintf(stderr, "FAIL: multi-attr device mismatch (%p, %zu, %d)\n",
                (void *)a_base, a_size, a_mtype);
        exit(1);
    }

    // --- unknown pointer returns INVALID_VALUE ---
    CUdeviceptr bogus = (CUdeviceptr)0xdeadbeef;
    CUresult err = cuPointerGetAttribute(&mtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, bogus);
    if (err != CUDA_ERROR_INVALID_VALUE) {
        fprintf(stderr, "FAIL: bogus ptr should be INVALID_VALUE, got %d\n", err);
        exit(1);
    }

    CHECK(cuMemFreeHost(hptr));
    CHECK(cuMemFree(dptr));
    CHECK(cuCtxDestroy(ctx));
    printf("ALL POINTER ATTR TESTS PASSED\n");
    return 0;
}

// stream_api_test.c: verify cuStreamGetFlags, cuStreamWaitEvent,
// cuStreamAddCallback, cuStreamGetPriority.
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

static int cb_called = 0;
static void *cb_userdata_seen = NULL;
static CUstream cb_stream_seen = NULL;

static void CUDA_CB my_callback(CUstream hStream, CUresult status, void *userData) {
    cb_called++;
    cb_stream_seen = hStream;
    cb_userdata_seen = userData;
    (void)status;
}

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUstream s1, s2;
    CUevent ev;
    CUdeviceptr d1 = 0, d2 = 0;
    int marker = 42;
    int *h = NULL;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CHECK(cuStreamCreate(&s1, 0));
    CHECK(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
    CHECK(cuEventCreate(&ev, 0));

    // --- cuStreamGetFlags ---
    unsigned int f1 = 0xdeadbeef;
    CHECK(cuStreamGetFlags(s1, &f1));
    if (f1 != 0) {
        fprintf(stderr, "FAIL: s1 flags %u != 0\n", f1);
        exit(1);
    }
    unsigned int f2 = 0xdeadbeef;
    CHECK(cuStreamGetFlags(s2, &f2));
    if (f2 != CU_STREAM_NON_BLOCKING) {
        fprintf(stderr, "FAIL: s2 flags %u != CU_STREAM_NON_BLOCKING(%d)\n",
                f2, CU_STREAM_NON_BLOCKING);
        exit(1);
    }

    // --- cuStreamGetPriority ---
    int prio = 0xdeadbeef;
    CHECK(cuStreamGetPriority(s1, &prio));
    if (prio != 0) {
        fprintf(stderr, "FAIL: priority %d != 0\n", prio);
        exit(1);
    }

    // --- cuStreamWaitEvent ---
    // Record an event on s1, wait on it from s2, then copy data on s2.
    // This exercises the wait-event barrier.
    CHECK(cuMemAlloc(&d1, sizeof(int)));
    CHECK(cuMemAlloc(&d2, sizeof(int)));
    CHECK(cuMemcpyHtoD(d1, &marker, sizeof(int)));
    CHECK(cuEventRecord(ev, s1));
    CHECK(cuStreamWaitEvent(s2, ev, 0));
    CHECK(cuMemcpyDtoDAsync(d2, d1, sizeof(int), s2));
    CHECK(cuStreamSynchronize(s2));
    h = malloc(sizeof(int));
    CHECK(cuMemcpyDtoH(h, d2, sizeof(int)));
    if (*h != marker) {
        fprintf(stderr, "FAIL: wait-event chain copied %d != %d\n", *h, marker);
        exit(1);
    }

    // --- cuStreamAddCallback ---
    // Callback fires after preceding work; stream must be empty or synced.
    CHECK(cuStreamSynchronize(s1));
    cb_called = 0;
    CHECK(cuStreamAddCallback(s1, my_callback, (void *)&marker, 0));
    if (cb_called != 1) {
        fprintf(stderr, "FAIL: callback not called (%d)\n", cb_called);
        exit(1);
    }
    if (cb_stream_seen != s1) {
        fprintf(stderr, "FAIL: callback stream mismatch\n");
        exit(1);
    }
    if (cb_userdata_seen != (void *)&marker) {
        fprintf(stderr, "FAIL: callback userdata mismatch\n");
        exit(1);
    }

    CHECK(cuMemFree(d1));
    CHECK(cuMemFree(d2));
    CHECK(cuEventDestroy(ev));
    CHECK(cuStreamDestroy(s1));
    CHECK(cuStreamDestroy(s2));
    CHECK(cuCtxDestroy(ctx));
    free(h);

    printf("ALL STREAM API TESTS PASSED\n");
    return 0;
}

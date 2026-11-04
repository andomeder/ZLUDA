// async_memcpy_test.c: verify cuMemcpyAsync/cuMemcpyHtoDAsync/cuMemcpyDtoHAsync/
// cuMemcpyDtoDAsync on a stream with events for ordering, plus sync variants.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                             \
    do {                                                                        \
        CUresult err = (call);                                                  \
        if (err != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, err);                                                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define N 256

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUevent start, end;
    CUdeviceptr d_src = 0, d_dst = 0;
    int *h_src = NULL, *h_dst = NULL;
    int i;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CHECK(cuStreamCreate(&stream, 0));
    CHECK(cuEventCreate(&start, 0));
    CHECK(cuEventCreate(&end, 0));

    h_src = malloc(N * sizeof(int));
    h_dst = malloc(N * sizeof(int));
    for (i = 0; i < N; i++) h_src[i] = i * 3 + 1;
    memset(h_dst, 0, N * sizeof(int));

    CHECK(cuMemAlloc(&d_src, N * sizeof(int)));
    CHECK(cuMemAlloc(&d_dst, N * sizeof(int)));

    // --- HtoD async on stream ---
    CHECK(cuMemcpyHtoDAsync(d_src, h_src, N * sizeof(int), stream));

    // --- DtoD async on stream (device copy) ---
    CHECK(cuMemcpyDtoDAsync(d_dst, d_src, N * sizeof(int), stream));

    // --- record event, then DtoH async after it ---
    CHECK(cuEventRecord(start, stream));
    CHECK(cuMemcpyDtoHAsync(h_dst, d_dst, N * sizeof(int), stream));
    CHECK(cuEventRecord(end, stream));

    // --- synchronous wait via event sync ---
    CHECK(cuEventSynchronize(end));

    for (i = 0; i < N; i++) {
        if (h_dst[i] != h_src[i]) {
            fprintf(stderr, "FAIL: async chain h_dst[%d]=%d != h_src[%d]=%d\n",
                    i, h_dst[i], i, h_src[i]);
            exit(1);
        }
    }

    float ms = 0.0f;
    CHECK(cuEventElapsedTime(&ms, start, end));

    // --- sync DtoD variant ---
    memset(h_dst, 0, N * sizeof(int));
    CHECK(cuMemcpyDtoD(d_dst, d_src, N * sizeof(int)));
    CHECK(cuMemcpyDtoH(h_dst, d_dst, N * sizeof(int)));
    for (i = 0; i < N; i++) {
        if (h_dst[i] != h_src[i]) {
            fprintf(stderr, "FAIL: sync DtoD h_dst[%d]=%d\n", i, h_dst[i]);
            exit(1);
        }
    }

    // --- plain cuMemcpyAsync (DtoD) on legacy stream ---
    CHECK(cuMemcpyAsync(d_dst, d_src, N * sizeof(int), 0));

    // --- verify stream query is ready after sync ---
    CHECK(cuStreamSynchronize(stream));
    CUresult q;
    CHECK(cuStreamQuery(stream));
    (void)q;

    CHECK(cuMemFree(d_src));
    CHECK(cuMemFree(d_dst));
    CHECK(cuEventDestroy(start));
    CHECK(cuEventDestroy(end));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuCtxDestroy(ctx));
    free(h_src);
    free(h_dst);

    printf("ALL ASYNC MEMCPY TESTS PASSED (elapsed %.3f ms)\n", ms);
    return 0;
}

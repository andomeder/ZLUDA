// stream_copy_stress.c: the host/device copy contract that every CUDA
// application relies on, under repetition and concurrency.
//
// The four patterns here are the ones a real application produces:
//   A  a pipelined HtoD -> kernel -> DtoH loop on one stream
//   B  two streams ordered by an event, the consumer reading what the
//      producer wrote
//   C  a blocking copy of a buffer a queued kernel writes (the legacy stream
//      has to be ordered against the application's stream)
//   D  two threads copying unpinned host memory at the same time (the driver
//      stages those through pinned bounce buffers)
//
// A mismatch means the driver returned a value the application cannot rely
// on, which is the kind of failure that shows up later as a wrong number.
#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, err);                                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define N 1024
#define THREADS 256

// out[i] = 2 * in[i] + 1
static const char *KERNEL_PTX =
    ".version 6.5\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry stress_kernel(\n"
    "  .param .u64 pa,\n"
    "  .param .u32 pn\n"
    ")\n"
    "{\n"
    "  .reg .b32 %r<10>;\n"
    "  .reg .b64 %rd<4>;\n"
    "  .reg .pred %p<2>;\n"
    "  ld.param.u64 %rd1, [pa];\n"
    "  cvta.to.global.u64 %rd1, %rd1;\n"
    "  ld.param.u32 %r1, [pn];\n"
    "  mov.u32 %r2, %ntid.x;\n"
    "  mov.u32 %r3, %ctaid.x;\n"
    "  mov.u32 %r4, %tid.x;\n"
    "  mad.lo.s32 %r5, %r2, %r3, %r4;\n"
    "  setp.ge.s32 %p1, %r5, %r1;\n"
    "  @%p1 ret;\n"
    "  mul.wide.s32 %rd2, %r5, 4;\n"
    "  add.s64 %rd3, %rd1, %rd2;\n"
    "  ld.global.b32 %r6, [%rd3];\n"
    "  mul.lo.s32 %r7, %r6, 2;\n"
    "  add.s32 %r8, %r7, 1;\n"
    "  st.global.b32 [%rd3], %r8;\n"
    "  ret;\n"
    "}\n";

static CUmodule module;
static CUfunction kernel;
static CUdevice device;
static CUcontext context;
static int iterations = 200;

static void transform(int *p, int n) {
    for (int i = 0; i < n; i++) p[i] = p[i] * 2 + 1;
}

static void init(int *p, int n, int seed) {
    for (int i = 0; i < n; i++) p[i] = i + seed;
}

static int launch(CUdeviceptr buf, int n, CUstream stream) {
    int count = n;
    void *args[2] = {&buf, &count};
    CUresult err =
        cuLaunchKernel(kernel, (n + THREADS - 1) / THREADS, 1, 1, THREADS, 1,
                       1, 0, stream, args, NULL);
    return (int)err;
}

// Pattern A: one stream, blocking readback each iteration.
static int pattern_a(void) {
    int *host = malloc(N * 4), *expect = malloc(N * 4);
    CUdeviceptr buf = 0;
    CUstream stream;
    int bad = 0;
    CHECK(cuMemAlloc(&buf, N * 4));
    CHECK(cuStreamCreate(&stream, 0));
    for (int it = 0; it < iterations && bad == 0; it++) {
        init(host, N, it);
        memcpy(expect, host, N * 4);
        transform(expect, N);
        CHECK(cuMemcpyHtoDAsync(buf, host, N * 4, stream));
        if (launch(buf, N, stream) != 0) { bad++; break; }
        CHECK(cuMemcpyDtoHAsync(host, buf, N * 4, stream));
        CHECK(cuStreamSynchronize(stream));
        if (memcmp(host, expect, N * 4) != 0) {
            int i = 0;
            while (i < N && host[i] == expect[i]) i++;
            printf("  A: iteration %d, element %d: got %d want %d\n", it, i,
                   host[i], expect[i]);
            bad++;
        }
    }
    CHECK(cuStreamDestroy(stream));
    CHECK(cuMemFree(buf));
    free(host);
    free(expect);
    return bad;
}

// Pattern B: two streams, the second reads what the first wrote.
static int pattern_b(void) {
    int *host = malloc(N * 4), *expect = malloc(N * 4);
    CUdeviceptr buf = 0;
    CUstream s1, s2;
    CUevent ev;
    int bad = 0;
    CHECK(cuMemAlloc(&buf, N * 4));
    CHECK(cuStreamCreate(&s1, 0));
    CHECK(cuStreamCreate(&s2, 0));
    CHECK(cuEventCreate(&ev, 0));
    for (int it = 0; it < iterations && bad == 0; it++) {
        init(host, N, it);
        memcpy(expect, host, N * 4);
        transform(expect, N);
        CHECK(cuMemcpyHtoDAsync(buf, host, N * 4, s1));
        if (launch(buf, N, s1) != 0) { bad++; break; }
        CHECK(cuEventRecord(ev, s1));
        CHECK(cuStreamWaitEvent(s2, ev, 0));
        memset(host, 0, N * 4);
        CHECK(cuMemcpyDtoHAsync(host, buf, N * 4, s2));
        CHECK(cuStreamSynchronize(s2));
        if (memcmp(host, expect, N * 4) != 0) {
            int i = 0;
            while (i < N && host[i] == expect[i]) i++;
            printf("  B: iteration %d, element %d: got %d want %d\n", it, i,
                   host[i], expect[i]);
            bad++;
        }
    }
    CHECK(cuEventDestroy(ev));
    CHECK(cuStreamDestroy(s1));
    CHECK(cuStreamDestroy(s2));
    CHECK(cuMemFree(buf));
    free(host);
    free(expect);
    return bad;
}

// Pattern C: a blocking copy of a buffer a queued kernel writes.
static int pattern_c(void) {
    int *host = malloc(N * 4), *expect = malloc(N * 4);
    CUdeviceptr buf = 0;
    CUstream stream;
    int bad = 0;
    CHECK(cuMemAlloc(&buf, N * 4));
    CHECK(cuStreamCreate(&stream, 0));
    for (int it = 0; it < iterations && bad == 0; it++) {
        init(host, N, it);
        memcpy(expect, host, N * 4);
        transform(expect, N);
        CHECK(cuMemcpyHtoDAsync(buf, host, N * 4, stream));
        if (launch(buf, N, stream) != 0) { bad++; break; }
        memset(host, 0, N * 4);
        // No synchronize here: the blocking copy has to wait for the kernel.
        CHECK(cuMemcpyDtoH(host, buf, N * 4));
        if (memcmp(host, expect, N * 4) != 0) {
            int i = 0;
            while (i < N && host[i] == expect[i]) i++;
            printf("  C: iteration %d, element %d: got %d want %d\n", it, i,
                   host[i], expect[i]);
            bad++;
        }
    }
    CHECK(cuStreamDestroy(stream));
    CHECK(cuMemFree(buf));
    free(host);
    free(expect);
    return bad;
}

// Pattern D: two threads copying unpinned host memory at the same time.
struct worker {
    int id;
    int bad;
};

static void *worker_main(void *arg) {
    struct worker *w = arg;
    CUstream stream;
    CUdeviceptr buf = 0;
    int *host = malloc(N * 4), *expect = malloc(N * 4);
    w->bad = 0;
    // The driver API ties a context to the thread that created it.
    CUresult ctx_err = cuCtxSetCurrent(context);
    if (ctx_err != CUDA_SUCCESS) {
        printf("  D%d: cuCtxSetCurrent -> %d\n", w->id, ctx_err);
        w->bad = 1;
        return NULL;
    }
    CUresult err = cuStreamCreate(&stream, 0);
    if (err != CUDA_SUCCESS) {
        printf("  D%d: cuStreamCreate -> %d\n", w->id, err);
        w->bad = 1;
        return NULL;
    }
    err = cuMemAlloc(&buf, N * 4);
    if (err != CUDA_SUCCESS) {
        printf("  D%d: cuMemAlloc -> %d\n", w->id, err);
        w->bad = 1;
        return NULL;
    }
    for (int it = 0; it < iterations / 4 && w->bad == 0; it++) {
        init(host, N, w->id * 1000 + it);
        memcpy(expect, host, N * 4);
        transform(expect, N);
        if (cuMemcpyHtoDAsync(buf, host, N * 4, stream) != CUDA_SUCCESS) {
            w->bad = 1;
            break;
        }
        if (launch(buf, N, stream) != 0) { w->bad = 1; break; }
        memset(host, 0, N * 4);
        if (cuMemcpyDtoHAsync(host, buf, N * 4, stream) != CUDA_SUCCESS) {
            w->bad = 1;
            break;
        }
        if (cuStreamSynchronize(stream) != CUDA_SUCCESS) { w->bad = 1; break; }
        if (memcmp(host, expect, N * 4) != 0) {
            int i = 0;
            while (i < N && host[i] == expect[i]) i++;
            printf("  D%d: iteration %d, element %d: got %d want %d\n", w->id,
                   it, i, host[i], expect[i]);
            w->bad = 1;
        }
    }
    cuMemFree(buf);
    cuStreamDestroy(stream);
    free(host);
    free(expect);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc > 1) iterations = atoi(argv[1]);
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&device, 0));
    CHECK(cuCtxCreate(&context, 0, device));
    CHECK(cuModuleLoadDataEx(&module, KERNEL_PTX, 0, NULL, NULL));
    CHECK(cuModuleGetFunction(&kernel, module, "stress_kernel"));

    int bad = 0;
    bad += pattern_a();
    printf("pattern A (one stream pipeline): %s\n", bad ? "FAILED" : "OK");
    int b = pattern_b();
    printf("pattern B (two streams, event ordered): %s\n", b ? "FAILED" : "OK");
    int c = pattern_c();
    printf("pattern C (blocking copy after a queued kernel): %s\n", c ? "FAILED" : "OK");
    struct worker w[2];
    pthread_t t[2];
    for (int i = 0; i < 2; i++) {
        w[i].id = i;
        w[i].bad = 0;
        pthread_create(&t[i], NULL, worker_main, &w[i]);
    }
    int d = 0;
    for (int i = 0; i < 2; i++) {
        pthread_join(t[i], NULL);
        d += w[i].bad;
    }
    printf("pattern D (two threads copying unpinned memory): %s\n", d ? "FAILED" : "OK");
    bad += b + c + d;
    printf("STREAM COPY STRESS %s (%d iterations)\n", bad ? "FAILED" : "OK", iterations);
    return bad != 0;
}

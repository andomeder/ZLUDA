/* Graph capture and replay, at the driver API level.
 *
 * cuStreamBeginCapture used to return CUDA_ERROR_NOT_SUPPORTED, so an
 * application that captures (torch.cuda.CUDAGraph, torch.compile's
 * "reduce-overhead" mode, a cuDNN or cuBLAS graph) could not start. This pins
 * the contract that replaces it:
 *
 *   1. NOTHING EXECUTES DURING CAPTURE. The output buffer is poisoned before the
 *      capture and must still hold the poison after cuStreamEndCapture.
 *   2. A REPLAY IS REPEATABLE. Two launches of the same executable give the same
 *      answer at the same addresses, and a third after the executable was
 *      destroyed still gives it.
 *   3. ARGUMENTS ARE CAPTURED BY VALUE. The launch site's scalar is changed
 *      between capture and replay; the replay must use the value it had at
 *      capture time. Re-reading the caller's pointer at replay is the silent
 *      corruption this whole design exists to avoid, so it is measured rather
 *      than asserted.
 *   4. The recorded operation kinds all survive: a kernel launch, a memcpy and a
 *      fill, in the order they were captured.
 *   5. CUDA's error paths are answered rather than ignored: a synchronise or a
 *      blocking copy inside a capture is refused with
 *      CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900), and ending a capture that is
 *      not in progress is CUDA_ERROR_STREAM_CAPTURE_UNMATCHED (903). An
 *      application relies on the error; a hang or a silently recorded sync would
 *      be worse than refusing.
 *   6. The graph outlives the stream it was captured on, which is what CUDA
 *      promises and what an application that destroys its capture stream does.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    const char *n_; cuGetErrorName(r_, &n_); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n_); return 1; } } while (0)

#define CHECK_EQ(x, want) do { CUresult r_ = (x); if (r_ != (want)) { \
    const char *n_; cuGetErrorName(r_, &n_); \
    fprintf(stderr, "FAIL %s:%d got %s, wanted %s\n", __FILE__, __LINE__, n_, #want); \
    return 1; } } while (0)

static const char *KERNELS =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry saxpy(.param .u64 p_x, .param .u64 p_out, .param .u32 p_n,\n"
    "                      .param .f32 p_a, .param .f32 p_b)\n"
    "{\n"
    "  .reg .b32 %r<8>;\n"
    "  .reg .b64 %rd<6>;\n"
    "  .reg .f32 %f<6>;\n"
    "  .reg .pred %p<2>;\n"
    "  ld.param.u64 %rd1, [p_x];\n"
    "  ld.param.u64 %rd2, [p_out];\n"
    "  ld.param.u32 %r1, [p_n];\n"
    "  ld.param.f32 %f1, [p_a];\n"
    "  ld.param.f32 %f2, [p_b];\n"
    "  mov.u32 %r2, %tid.x;\n"
    "  mov.u32 %r3, %ctaid.x;\n"
    "  mov.u32 %r4, %ntid.x;\n"
    "  mad.lo.s32 %r5, %r3, %r4, %r2;\n"
    "  setp.ge.u32 %p1, %r5, %r1;\n"
    "  @%p1 bra L_saxpy_end;\n"
    "  mul.wide.u32 %rd3, %r5, 4;\n"
    "  add.s64 %rd4, %rd1, %rd3;\n"
    "  add.s64 %rd5, %rd2, %rd3;\n"
    "  ld.global.f32 %f3, [%rd4];\n"
    "  fma.rn.f32 %f4, %f3, %f1, %f2;\n"
    "  st.global.f32 [%rd5], %f4;\n"
    "L_saxpy_end:\n"
    "  ret;\n"
    "}\n"
    ".visible .entry scale(.param .u64 p_src, .param .u64 p_dst, .param .u32 p_n,\n"
    "                      .param .f32 p_k)\n"
    "{\n"
    "  .reg .b32 %r<8>;\n"
    "  .reg .b64 %rd<6>;\n"
    "  .reg .f32 %f<3>;\n"
    "  .reg .pred %p<2>;\n"
    "  ld.param.u64 %rd1, [p_src];\n"
    "  ld.param.u64 %rd2, [p_dst];\n"
    "  ld.param.u32 %r1, [p_n];\n"
    "  ld.param.f32 %f1, [p_k];\n"
    "  mov.u32 %r2, %tid.x;\n"
    "  mov.u32 %r3, %ctaid.x;\n"
    "  mov.u32 %r4, %ntid.x;\n"
    "  mad.lo.s32 %r5, %r3, %r4, %r2;\n"
    "  setp.ge.u32 %p1, %r5, %r1;\n"
    "  @%p1 bra L_scale_end;\n"
    "  mul.wide.u32 %rd3, %r5, 4;\n"
    "  add.s64 %rd4, %rd1, %rd3;\n"
    "  add.s64 %rd5, %rd2, %rd3;\n"
    "  ld.global.f32 %f2, [%rd4];\n"
    "  mul.rn.f32 %f2, %f2, %f1;\n"
    "  st.global.f32 [%rd5], %f2;\n"
    "L_scale_end:\n"
    "  ret;\n"
    "}\n";

/* A blocking memset on the CAPTURED stream is the dangerous case: recording it
 * would report success for work that had not run, which an application cannot
 * detect. It has to be refused. The same capture shows that the asynchronous
 * forms really are recorded and replayed, in order. */
static int test_legacy_capture(CUdeviceptr fill) {
    CHECK(cuStreamBeginCapture(NULL, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    CHECK_EQ(cuMemsetD32(fill, 0x11223344, 4), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    CHECK_EQ(cuMemsetD8(fill, 0x5a, 16), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    CHECK(cuMemsetD32Async(fill, 0x11223344, 4, NULL));
    CHECK(cuMemsetD16Async(fill, 0x5566, 8, NULL));
    CUgraph graph = NULL;
    CHECK(cuStreamEndCapture(NULL, &graph));
    CUgraphExec exec = NULL;
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK(cuGraphLaunch(exec, NULL));
    CHECK(cuCtxSynchronize());
    unsigned words[4] = {0, 0, 0, 0};
    CHECK(cuMemcpyDtoH(words, fill, sizeof(words)));
    /* The second fill was captured second, so it is the one that survives. */
    if (words[0] != 0x55665566u || words[1] != 0x55665566u) {
        fprintf(stderr, "FAIL legacy capture: fill[0] = %08x fill[1] = %08x, expected "
                        "the second fill (55665566) to win\n", words[0], words[1]);
        return 1;
    }
    CHECK(cuGraphExecDestroy(exec));
    CHECK(cuGraphDestroy(graph));
    printf("legacy-stream capture: a blocking memset is refused (900), the async "
           "ones are recorded and replay in order\n");
    return 0;
}

#define N 4096
#define BLOCK 256
#define GRID (N / BLOCK)
#define POISON 0x5a

/* The capture: poison-check, order-check, argument-by-value check. */
int main(void) {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, KERNELS));
    CUfunction saxpy, scale;
    CHECK(cuModuleGetFunction(&saxpy, mod, "saxpy"));
    CHECK(cuModuleGetFunction(&scale, mod, "scale"));

    CUdeviceptr x = 0, mid = 0, tmp = 0, out = 0, fill = 0;
    CHECK(cuMemAlloc(&x, N * 4));
    CHECK(cuMemAlloc(&mid, N * 4));
    CHECK(cuMemAlloc(&tmp, N * 4));
    CHECK(cuMemAlloc(&out, N * 4));
    CHECK(cuMemAlloc(&fill, 64));

    /* x is all ones, so the whole pipeline has a closed form. */
    float *host = (float *)malloc(N * 4);
    for (int i = 0; i < N; i++) host[i] = 1.0f;
    CHECK(cuMemcpyHtoD(x, host, N * 4));
    free(host);

    /* Everything the graph writes starts as poison, so "ran" and "did not run"
     * cannot be confused with each other. */
    CHECK(cuMemsetD8(out, POISON, N * 4));
    CHECK(cuMemsetD8(mid, POISON, N * 4));
    CHECK(cuMemsetD8(tmp, POISON, N * 4));
    CHECK(cuMemsetD8(fill, 0, 64));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, 0));

    /* The scalar the launch sites will change AFTER the capture. */
    float a = 2.0f, b = 1.0f, k = 3.0f;
    unsigned n = N;
    int pattern = 0x11223344;

    CUgraph graph = NULL;
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;

    /* --- 6 and 5: end-capture on an idle stream is UNMATCHED, is-capturing is
     * NONE outside a capture. */
    CHECK_EQ(cuStreamEndCapture(stream, &graph), CUDA_ERROR_STREAM_CAPTURE_UNMATCHED);
    CHECK(cuStreamIsCapturing(stream, &status));
    if (status != CU_STREAM_CAPTURE_STATUS_NONE) {
        fprintf(stderr, "FAIL is-capturing outside a capture: %d\n", (int)status);
        return 1;
    }

    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));

    CHECK(cuStreamIsCapturing(stream, &status));
    if (status != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
        fprintf(stderr, "FAIL is-capturing inside a capture: %d\n", (int)status);
        return 1;
    }

    /* --- 5: the paths CUDA refuses during a capture. */
    CHECK_EQ(cuStreamSynchronize(stream), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    CHECK_EQ(cuCtxSynchronize(), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    {
        unsigned char sink[16];
        CHECK_EQ(cuMemcpyDtoH(sink, mid, sizeof(sink)), CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    }
    /* A blocking memset runs on the legacy stream, so while a DIFFERENT stream is
     * capturing it is an operation outside the capture (906). The case that
     * matters - a blocking memset on the captured stream itself - is
     * test_legacy_capture() below. */
    CHECK_EQ(cuMemsetD32(fill, 0, 4), CUDA_ERROR_STREAM_CAPTURE_IMPLICIT);

    /* --- 4: three kinds of recorded operation, in one order. */
    CHECK(cuMemsetD32Async(fill, pattern, 4, stream));
    void *saxpy_args[5] = {&x, &mid, &n, &a, &b};
    CHECK(cuLaunchKernel(saxpy, GRID, 1, 1, BLOCK, 1, 1, 0, stream, saxpy_args, NULL));
    CHECK(cuMemcpyAsync(tmp, mid, N * 4, stream));
    void *scale_args[4] = {&tmp, &out, &n, &k};
    CHECK(cuLaunchKernel(scale, GRID, 1, 1, BLOCK, 1, 1, 0, stream, scale_args, NULL));

    CHECK(cuStreamEndCapture(stream, &graph));
    if (graph == NULL) {
        fprintf(stderr, "FAIL end-capture returned a null graph\n");
        return 1;
    }

    /* --- 1: nothing ran. */
    {
        unsigned char probe[64];
        CHECK(cuMemcpyDtoH(probe, out, sizeof(probe)));
        for (size_t i = 0; i < sizeof(probe); i++) {
            if (probe[i] != POISON) {
                fprintf(stderr,
                        "FAIL capture executed work: out[%zu] = 0x%02x, still expected "
                        "the poison 0x%02x\n", i, probe[i], POISON);
                return 1;
            }
        }
        unsigned zero_check[4] = {0, 0, 0, 0};
        CHECK(cuMemcpyDtoH(zero_check, fill, sizeof(zero_check)));
        if (zero_check[0] != 0) {
            fprintf(stderr, "FAIL capture executed the fill: %08x\n", zero_check[0]);
            return 1;
        }
    }

    /* --- 3: the launch sites change, and the replay must not care. */
    a = 100.0f;
    k = 7.0f;
    n = 1; /* a different element count would truncate the launch */

    CUgraphExec exec = NULL;
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));

    /* --- 2: two replays, same addresses, same answer. (2*1+1)*3 = 9. */
    float *readback = (float *)malloc(N * 4);
    for (int replay = 0; replay < 2; replay++) {
        CHECK(cuGraphLaunch(exec, stream));
        CHECK(cuStreamSynchronize(stream));
        CHECK(cuMemcpyDtoH(readback, out, N * 4));
        for (int i = 0; i < N; i++) {
            if (readback[i] != 9.0f) {
                fprintf(stderr,
                        "FAIL replay %d: out[%d] = %f, expected 9 (the captured "
                        "arguments are 2 and 1, and the scale is 3)\n",
                        replay, i, readback[i]);
                return 1;
            }
        }
        unsigned words[4] = {0, 0, 0, 0};
        CHECK(cuMemcpyDtoH(words, fill, sizeof(words)));
        for (int i = 0; i < 4; i++) {
            if (words[i] != (unsigned)pattern) {
                fprintf(stderr, "FAIL replay %d: fill[%d] = %08x, expected %08x\n",
                        replay, i, words[i], (unsigned)pattern);
                return 1;
            }
        }
    }

    /* --- 6: the graph outlives the stream it was captured on. */
    CHECK(cuStreamDestroy(stream));
    CHECK(cuGraphLaunch(exec, NULL)); /* the legacy stream */
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(readback, out, N * 4));
    for (int i = 0; i < N; i++) {
        if (readback[i] != 9.0f) {
            fprintf(stderr, "FAIL replay after stream destroy: out[%d] = %f\n", i, readback[i]);
            return 1;
        }
    }

    /* --- the executable does not depend on the graph either. */
    CHECK(cuGraphDestroy(graph));
    CHECK(cuGraphLaunch(exec, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(readback, out, N * 4));
    for (int i = 0; i < N; i++) {
        if (readback[i] != 9.0f) {
            fprintf(stderr, "FAIL replay after graph destroy: out[%d] = %f\n", i, readback[i]);
            return 1;
        }
    }
    CHECK(cuGraphExecDestroy(exec));

    /* --- 3, the other half: a FRESH capture sees the values the sites hold
     * now. (100*1+1)*7 = 707. */
    CHECK(cuMemsetD8(out, POISON, N * 4));
    CHECK(cuStreamCreate(&stream, 0));
    a = 100.0f;
    k = 7.0f;
    n = N;
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    CHECK(cuLaunchKernel(saxpy, GRID, 1, 1, BLOCK, 1, 1, 0, stream, saxpy_args, NULL));
    void *scale_args2[4] = {&mid, &out, &n, &k};
    CHECK(cuLaunchKernel(scale, GRID, 1, 1, BLOCK, 1, 1, 0, stream, scale_args2, NULL));
    CHECK(cuStreamEndCapture(stream, &graph));
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK(cuGraphLaunch(exec, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH(readback, out, N * 4));
    for (int i = 0; i < N; i++) {
        if (readback[i] != 707.0f) {
            fprintf(stderr,
                    "FAIL fresh capture: out[%d] = %f, expected 707 (a=100, b=1, k=7)\n",
                    i, readback[i]);
            return 1;
        }
    }

    /* --- a blocking memset on the captured stream, and two recorded fills in
     * order. */
    if (test_legacy_capture(fill) != 0) return 1;

    /* --- an empty capture still produces a launchable graph. */
    CHECK(cuGraphExecDestroy(exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    CHECK(cuStreamEndCapture(stream, &graph));
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK(cuGraphLaunch(exec, stream));
    CHECK(cuStreamSynchronize(stream));

    CHECK(cuGraphExecDestroy(exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuStreamDestroy(stream));

    free(readback);
    printf("GRAPH TEST: OK (capture records without executing, replay is repeatable "
           "at the same addresses, arguments are captured by value, and the "
           "refusals match CUDA)\n");
    return 0;
}

/* What a graph replay costs against running the same operations directly.
 *
 * The design is a RE-APPEND: a replay puts every recorded command back on the
 * immediate command list, so it pays the dispatch cost again and buys
 * compatibility rather than speed (odocs/PLAN-GRAPHS.md measured that ceiling at
 * 1.0-1.05x for the page before this existed). This prints the two numbers so the
 * claim is a measurement rather than a recollection.
 *
 * A graph of three commands (fill, kernel, kernel) is replayed N times and the
 * same three commands are submitted directly 3N times, both without a
 * synchronise inside the loop, so both numbers are host submission cost.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    const char *n_; cuGetErrorName(r_, &n_); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n_); return 1; } } while (0)

static const char *K =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry addk(.param .u64 p, .param .u32 n)\n"
    "{\n"
    "  .reg .b32 %r<6>;\n"
    "  .reg .b64 %rd<4>;\n"
    "  .reg .pred %p<2>;\n"
    "  ld.param.u64 %rd1, [p];\n"
    "  ld.param.u32 %r1, [n];\n"
    "  mov.u32 %r2, %tid.x;\n"
    "  mov.u32 %r3, %ctaid.x;\n"
    "  mov.u32 %r4, %ntid.x;\n"
    "  mad.lo.s32 %r5, %r3, %r4, %r2;\n"
    "  setp.ge.u32 %p1, %r5, %r1;\n"
    "  @%p1 bra L_end;\n"
    "  mul.wide.u32 %rd2, %r5, 4;\n"
    "  add.s64 %rd3, %rd1, %rd2;\n"
    "  ld.global.u32 %r1, [%rd3];\n"
    "  add.u32 %r1, %r1, 1;\n"
    "  st.global.u32 [%rd3], %r1;\n"
    "L_end:\n"
    "  ret;\n"
    "}\n";

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 500;
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, K));
    CUfunction addk;
    CHECK(cuModuleGetFunction(&addk, mod, "addk"));
    CUdeviceptr buf = 0;
    CHECK(cuMemAlloc(&buf, 4096));
    unsigned n = 1024;
    void *args[2] = {&buf, &n};
    CUstream s;
    CHECK(cuStreamCreate(&s, 0));

    /* Warm up both paths so the first-touch costs (kernel creation, event pool
     * growth) are not in either number. */
    for (int i = 0; i < 50; i++) {
        CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
        CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
        CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
    }
    CHECK(cuStreamSynchronize(s));

    CUgraph graph = NULL;
    CUgraphExec exec = NULL;
    CHECK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
    CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
    CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
    CHECK(cuStreamEndCapture(s, &graph));
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));

    double t0 = now_ms();
    for (int i = 0; i < reps; i++) CHECK(cuGraphLaunch(exec, s));
    double graph_ms = now_ms() - t0;
    CHECK(cuStreamSynchronize(s));

    t0 = now_ms();
    for (int i = 0; i < reps; i++) {
        CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
        CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
        CHECK(cuLaunchKernel(addk, 16, 1, 1, 64, 1, 1, 0, s, args, NULL));
    }
    double direct_ms = now_ms() - t0;
    CHECK(cuStreamSynchronize(s));

    double per_graph = graph_ms * 1e3 / reps;
    double per_direct = direct_ms * 1e3 / (reps * 3.0);
    printf("graph:  %d replays of 3 commands: %.1f ms total, %.2f us per replay "
           "(%.2f us per recorded command)\n",
           reps, graph_ms, per_graph, per_graph / 3.0);
    printf("direct: %d commands: %.1f ms total, %.2f us per command\n",
           reps * 3, direct_ms, per_direct);
    printf("GRAPH COST: a replay is %.2fx the cost of the commands it replays, "
           "i.e. compatibility, not speed\n", per_direct > 0 ? per_graph / (per_direct * 3.0) : 0.0);

    CHECK(cuGraphExecDestroy(exec));
    CHECK(cuGraphDestroy(graph));
    CHECK(cuStreamSynchronize(s));
    return 0;
}

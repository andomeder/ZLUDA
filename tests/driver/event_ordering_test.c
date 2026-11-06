/* A recorded event must not report ready before the work queued ahead of it has
 * retired.
 *
 * cuEventRecord appends a barrier to the stream. That barrier is ordered behind
 * everything already queued because the command list is in order - but the
 * stream's event recycler does not know about it, so if the barrier also waits
 * on a stream event, that event can be reset while the barrier is still pending.
 * The symptom is an event that either never fires or fires early, and an early
 * event lets a caller free a buffer a kernel is still reading.
 *
 * This launches work that takes long enough to observe, records an event behind
 * it, and checks that the event is NOT ready until the work retires.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *n; cuGetErrorName(r, &n); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n); return 1; } } while (0)

/* A grid-stride loop over a large buffer: long enough that the queue is still
 * busy when the event is recorded behind it. */
static const char *SPIN_PTX =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry spin(.param .u64 out, .param .u32 rounds)\n"
    "{\n"
    "  .reg .pred %p1;\n"
    "  .reg .b32 %r<8>;\n"
    "  .reg .b64 %rd<8>;\n"
    "  ld.param.u64 %rd1, [out];\n"
    "  ld.param.u32 %r1, [rounds];\n"
    "  /* each thread owns its own word, so the loop measures work and not\n"
    "     contention on a single address */\n"
    "  mov.u32 %r2, %tid.x;\n"
    "  mov.u32 %r4, %ctaid.x;\n"
    "  mul.lo.u32 %r5, %r4, 256;\n"
    "  add.u32 %r2, %r2, %r5;\n"
    "  mul.wide.u32 %rd2, %r2, 4;\n"
    "  add.s64 %rd3, %rd1, %rd2;\n"
    "  mov.u32 %r3, 0;\n"
    "L_loop:\n"
    "  setp.ge.u32 %p1, %r3, %r1;\n"
    "  @%p1 bra L_done;\n"
    "  ld.global.u32 %r6, [%rd3];\n"
    "  add.u32 %r6, %r6, 1;\n"
    "  st.global.u32 [%rd3], %r6;\n"
    "  add.u32 %r3, %r3, 1;\n"
    "  bra L_loop;\n"
    "L_done:\n"
    "  ret;\n"
    "}\n";

int main(void) {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, SPIN_PTX));
    CUfunction spin;
    CHECK(cuModuleGetFunction(&spin, mod, "spin"));
    CUstream stream;
    CHECK(cuStreamCreate(&stream, 0));
    CUdeviceptr out;
    CHECK(cuMemAlloc(&out, 8 * 256 * 4));
    CHECK(cuMemsetD32(out, 0, 8 * 256));

    unsigned rounds = 20000;
    void *args[2] = {&out, &rounds};
    CUevent event;
    CHECK(cuEventCreate(&event, 0));

    int early = 0;
    int late = 0;
    for (int trial = 0; trial < 5; trial++) {
        /* Four long kernels, then an event behind them. */
        for (int i = 0; i < 4; i++) {
            CHECK(cuLaunchKernel(spin, 8, 1, 1, 256, 1, 1, 0, stream, args, NULL));
        }
        CHECK(cuEventRecord(event, stream));
        /* The event must not be ready while the kernels are still running. */
        CUresult query = cuEventQuery(event);
        if (query == CUDA_SUCCESS) {
            early++;
        }
        CHECK(cuEventSynchronize(event));
        if (cuEventQuery(event) != CUDA_SUCCESS) {
            late++;
        }
    }

    long long elapsed_checks = 0;
    (void)elapsed_checks;
    printf("EVENT ORDERING TEST: %d early of 5, %d not ready after sync of 5\n", early, late);

    CHECK(cuEventDestroy(event));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuMemFree(out));
    if (early || late) {
        printf("EVENT ORDERING TEST: FAILED\n");
        return 1;
    }
    printf("EVENT ORDERING TEST: OK\n");
    return 0;
}

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
 * THE WORK HERE CANNOT RETIRE. The kernel spins on a word in pinned host memory
 * until this program writes it, so while the event is recorded the work is still
 * running by construction and the check is a measurement rather than a race.
 * That matters: this test used to put four increment loops in front of the event
 * and call them long, but IGC folds such a loop into a single add, so the four
 * kernels were microseconds and the event was legitimately ready by the time the
 * host asked - which is why it reported "1 early of 5" in some machine states
 * and not in others (measured: 2000 launches of that kernel take 65 ms at
 * rounds=0 and 73 ms at rounds=2,000,000, and an LCG chain of the same shape,
 * which cannot be folded, costs 70x more per launch).
 *
 * The spin kernel also bumps a counter after it leaves the loop, so "the work
 * really ran" is checked rather than assumed, and a launch that silently did
 * nothing cannot make this test pass.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *n; cuGetErrorName(r, &n); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n); return 1; } } while (0)

/* One thread per output word spins on the host word, then thread 0 of each block
 * adds one to the device counter. The iteration cap is a safety net so a lost
 * release cannot wedge the card; it is several seconds of volatile loads, far
 * longer than any window this test waits for, so a cap that fires shows up as a
 * failure rather than as a hang. */
static const char *SPIN_PTX =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry spin(.param .u64 release, .param .u64 counter)\n"
    ".maxntid 256, 1, 1\n"
    "{\n"
    "  .reg .pred %p1;\n"
    "  .reg .pred %p2;\n"
    "  .reg .b32 %r<6>;\n"
    "  .reg .b64 %rd<4>;\n"
    "  ld.param.u64 %rd1, [release];\n"
    "  ld.param.u64 %rd2, [counter];\n"
    "  mov.u32 %r2, 0;\n"
    "L_spin:\n"
    "  ld.volatile.global.u32 %r1, [%rd1];\n"
    "  setp.ne.u32 %p1, %r1, 0;\n"
    "  @%p1 bra L_done;\n"
    "  add.u32 %r2, %r2, 1;\n"
    "  setp.lt.u32 %p2, %r2, 40000000;\n"
    "  @%p2 bra L_spin;\n"
    "L_done:\n"
    "  mov.u32 %r3, %tid.x;\n"
    "  setp.ne.u32 %p1, %r3, 0;\n"
    "  @%p1 bra L_exit;\n"
    "  atom.global.add.u32 %r4, [%rd2], 1;\n"
    "L_exit:\n"
    "  ret;\n"
    "}\n";

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define TRIALS 5
#define LAUNCHES_PER_TRIAL 4
#define GRID 8
#define POLL_MS 25.0

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

    /* The release word lives in pinned host memory: the kernel reads it and the
     * host writes it with a plain store, so releasing the spin needs no driver
     * call that could queue behind the kernel it is releasing. */
    volatile unsigned *release_host = NULL;
    CHECK(cuMemAllocHost((void **)&release_host, 4096));
    *release_host = 0;
    CUdeviceptr release_dev = 0;
    CHECK(cuMemHostGetDevicePointer_v2(&release_dev, (void *)release_host, 0));

    CUdeviceptr counter = 0;
    CHECK(cuMemAlloc(&counter, 8));
    CHECK(cuMemsetD32(counter, 0, 2));

    void *args[2] = {&release_dev, &counter};
    CUevent event;
    CHECK(cuEventCreate(&event, 0));

    int early = 0;
    int late = 0;
    for (int trial = 0; trial < TRIALS; trial++) {
        *release_host = 0;
        for (int i = 0; i < LAUNCHES_PER_TRIAL; i++) {
            CHECK(cuLaunchKernel(spin, GRID, 1, 1, 256, 1, 1, 0, stream, args, NULL));
        }
        CHECK(cuEventRecord(event, stream));
        /* The kernels are spinning, so they cannot have retired however long
         * the driver takes to get the barrier executed. Any readiness inside
         * this window means the event answered before its work did. */
        double t0 = now_ms();
        while (now_ms() - t0 < POLL_MS) {
            if (cuEventQuery(event) == CUDA_SUCCESS) {
                early++;
                break;
            }
        }
        /* Let the work finish and require the event to fire. */
        *release_host = 1;
        CHECK(cuEventSynchronize(event));
        if (cuEventQuery(event) != CUDA_SUCCESS) {
            late++;
        }
    }

    unsigned ran = 0;
    CHECK(cuMemcpyDtoH(&ran, counter, sizeof(ran)));
    unsigned want = TRIALS * LAUNCHES_PER_TRIAL * GRID;
    printf("EVENT ORDERING TEST: %d early of %d, %d not ready after sync of %d, "
           "work %u/%u %s\n", early, TRIALS, late, TRIALS, ran, want,
           ran == want ? "OK" : "MISMATCH");

    CHECK(cuEventDestroy(event));
    CHECK(cuStreamDestroy(stream));
    CHECK(cuMemFree(counter));
    CHECK(cuMemFreeHost((void *)release_host));
    if (early || late || ran != want) {
        printf("EVENT ORDERING TEST: FAILED\n");
        return 1;
    }
    printf("EVENT ORDERING TEST: OK\n");
    return 0;
}

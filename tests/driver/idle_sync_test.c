/* A synchronize on a stream with nothing queued must return at once.
 *
 * It used to append a barrier with an empty wait list, and a barrier with no
 * wait events does not signal on this driver, so the sync waited out the whole
 * timeout with no work pending. Any application that syncs an idle stream paid
 * 60 seconds for it, and a decode loop does it between steps.
 *
 * ZLUDA_IDLE_SYNC_BARRIER=1 restores the old behaviour, which is how this test
 * is shown to be able to fail.
 */
#include <cuda.h>
#include <stdio.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *n; cuGetErrorName(r, &n); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n); return 1; } } while (0)

int main(void) {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    /* Nothing has been launched, so the stream is idle. */
    double started = now_ms();
    for (int i = 0; i < 5; i++) {
        CHECK(cuCtxSynchronize());
    }
    double idle = now_ms() - started;

    /* Now with one launch queued and retired, then idle again. */
    CUstream stream;
    CHECK(cuStreamCreate(&stream, 0));
    CHECK(cuStreamSynchronize(stream));
    started = now_ms();
    for (int i = 0; i < 5; i++) {
        CHECK(cuStreamSynchronize(stream));
    }
    double empty_stream = now_ms() - started;
    CHECK(cuStreamDestroy(stream));

    printf("IDLE SYNC TEST: ctx %.1f ms, empty stream %.1f ms for 5 syncs\n",
           idle, empty_stream);
    /* Generous: each stalled sync costs the 60 s timeout. */
    if (idle > 2000.0 || empty_stream > 2000.0) {
        printf("IDLE SYNC TEST: FAILED (a sync on idle work is stalling)\n");
        return 1;
    }
    printf("IDLE SYNC TEST: OK\n");
    return 0;
}

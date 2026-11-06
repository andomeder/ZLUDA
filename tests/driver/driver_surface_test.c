/* The driver surface an application touches but which does not change what runs.
 *
 * These entry points were unimplemented stubs: context limits, stream
 * attributes, managed-memory attach, memory-pool trimming. None of them changes
 * what a stream executes, so the honest behaviour is to accept the value and
 * report it back, but a stub that fails is worse than a no-op: an application
 * that checks the result aborts, and the abort looks like a driver gap rather
 * than "this call has no effect here".
 *
 * cuLaunchHostFunc is the exception that does have an effect: the callback has to
 * run after the work queued ahead of it.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;

#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *n; cuGetErrorName(r, &n); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n); failures++; } } while (0)

static int callback_ran;
static void host_callback(void *user_data) {
    callback_ran = *(int *)user_data;
}

static const char *FILL_PTX =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry fill(.param .u64 out, .param .u32 rounds)\n"
    "{\n"
    "  .reg .pred %p1;\n"
    "  .reg .b32 %r<8>;\n"
    "  .reg .b64 %rd<8>;\n"
    "  ld.param.u64 %rd1, [out];\n"
    "  ld.param.u32 %r1, [rounds];\n"
    "  mov.u32 %r3, 0;\n"
    "L_loop:\n"
    "  setp.ge.u32 %p1, %r3, %r1;\n"
    "  @%p1 bra L_done;\n"
    "  ld.global.u32 %r6, [%rd1];\n"
    "  add.u32 %r6, %r6, 1;\n"
    "  st.global.u32 [%rd1], %r6;\n"
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

    /* Context limits: accept what is set and report it back. */
    CHECK(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 2 << 20));
    size_t got = 0;
    CHECK(cuCtxGetLimit(&got, CU_LIMIT_PRINTF_FIFO_SIZE));
    printf("  printf fifo limit set 2 MiB, read back %zu\n", got);
    if (got != (2u << 20)) {
        printf("  FAIL: limit did not round-trip\n");
        failures++;
    }
    CUfunc_cache cache;
    CHECK(cuCtxGetCacheConfig(&cache));

    CUstream stream;
    CHECK(cuStreamCreate(&stream, 0));

    /* Stream attributes: advisory, so setting one must not fail. */
    CUstreamAttrValue value;
    CHECK(cuStreamSetAttribute(stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &value));
    CUstreamAttrValue readback;
    CHECK(cuStreamGetAttribute(stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &readback));
    CHECK(cuStreamCopyAttributes(stream, stream));
    CHECK(cuStreamAttachMemAsync(stream, 0, 0, CU_MEM_ATTACH_GLOBAL));

    /* Memory pool calls: allocations are not pooled, but they must not fail. */
    CUmemoryPool pool = 0;
    CHECK(cuDeviceGetDefaultMemPool(&pool, dev));
    CHECK(cuMemPoolTrimTo(pool, 0));

    /* A host function has to run after the work queued ahead of it. */
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, FILL_PTX));
    CUfunction fill;
    CHECK(cuModuleGetFunction(&fill, mod, "fill"));
    CUdeviceptr out;
    CHECK(cuMemAlloc(&out, 4));
    CHECK(cuMemsetD32(out, 0, 1));
    unsigned rounds = 20000;
    void *args[2] = {&out, &rounds};
    CHECK(cuLaunchKernel(fill, 1, 1, 1, 1, 1, 1, 0, stream, args, NULL));
    int marker = 42;
    CHECK(cuLaunchHostFunc(stream, host_callback, &marker));
    unsigned int seen = 0;
    CHECK(cuMemcpyDtoH(&seen, out, 4));
    printf("  host function ran: %d, value after the queued kernel: %u\n", callback_ran, seen);
    if (!callback_ran) {
        printf("  FAIL: the host function never ran\n");
        failures++;
    }
    if (seen != rounds) {
        printf("  FAIL: the callback ran before the kernel finished (%u of %u)\n", seen, rounds);
        failures++;
    }

    CHECK(cuStreamDestroy(stream));
    CHECK(cuMemFree(out));
    if (failures) {
        printf("DRIVER SURFACE TEST: %d failures\n", failures);
        return 1;
    }
    printf("DRIVER SURFACE TEST: OK\n");
    return 0;
}

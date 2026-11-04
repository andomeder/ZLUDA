// bigarg_test.c - launch a kernel whose by-value argument list exceeds the
// 2048 byte total that IGC accepts for a kernel signature.
//
// bigarg_kernel(out, blob[3000] by value, n): out[tid] = u32 at byte offset
// 8 + (tid % 8) * 4 of the block for tid < n, and 0 past n. The block is
// filled with 0xB0000000 + i at 8 + i*4, so every written word must read
// 0xB0000000 + (tid % 8): a wrong block layout, a wrong parameter offset or a
// dropped argument is visible in the value, not only in the launch result.
//
// Expected on the current driver: the argument list is 8 + 3000 + 4 = 3012
// bytes, IGC rejects the build with "Total size of kernel arguments exceeds
// limit!" and the module load fails. The harness reports that as the expected
// baseline rather than as a test error. Once the parameter block is passed
// indirectly (the argument bytes copied to a device buffer and its address
// passed in their place, as CUDA does for oversized signatures) this test has
// to move to the 256/256 branch.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <stdint.h>

#define CHECK(x) do { \
    CUresult r = (x); \
    if (r != CUDA_SUCCESS) { \
        const char* name; const char* str; \
        cuGetErrorName(r, &name); \
        cuGetErrorString(r, &str); \
        fprintf(stderr, "FAIL at %s:%d: %s (%s)\n", __FILE__, __LINE__, name, str); \
        exit(1); \
    } \
} while (0)

static void print_result(const char* what, CUresult r) {
    const char* name = "?"; const char* str = "?";
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &str);
    printf("%s: %s (%s), code %d\n", what, name, str, (int)r);
}

int main(void) {
    enum { NOUT = 256, NWORDS = 8, BLOB_SIZE = 3000, NTHREADS = 256, N = 200 };
    const uint32_t base = 0xB0000000u;
    const uint32_t poison = 0xDEADBEEFu;

    uint8_t blob[BLOB_SIZE];
    uint32_t out[NOUT];

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction kernel;
    CUdeviceptr d_out;

    memset(blob, 0, sizeof(blob));
    for (int i = 0; i < NWORDS; i++) {
        uint32_t v = base + (uint32_t)i;
        memcpy(blob + 8 + i * 4, &v, sizeof(v));
    }
    for (int i = 0; i < NOUT; i++) out[i] = poison;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));

    FILE* f = fopen("bigarg_test.ptx", "rb");
    if (!f) { perror("bigarg_test.ptx"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1);
    if (!ptx || fread(ptx, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 1; }
    ptx[sz] = 0;
    fclose(f);

    printf("arguments: out (8 bytes) + blob (3000 bytes by value) + n (4 bytes)"
           " = 3012 bytes, IGC limit 2048\n");

    CUresult r = cuModuleLoadData(&mod, ptx);
    if (r != CUDA_SUCCESS) {
        printf("BIGARG TEST: module build rejected the argument list "
               "(expected before the parameter buffer exists): %d\n", (int)r);
        print_result("module load", r);
        cuCtxDestroy(ctx);
        free(ptx);
        return 2;
    }
    printf("module load OK\n");

    // The driver builds a module lazily, so an IGC rejection can surface at
    // either the load or the function lookup. Both are the expected baseline
    // here, and the message says which one it was.
    r = cuModuleGetFunction(&kernel, mod, "bigarg_kernel");
    if (r != CUDA_SUCCESS) {
        printf("BIGARG TEST: module build rejected the argument list "
               "(expected before the parameter buffer exists): %d\n", (int)r);
        print_result("cuModuleGetFunction", r);
        printf("BIGARG TEST: IGC build log for this signature: \"error: in kernel "
               "'bigarg_kernel': Total size of kernel arguments exceeds limit! "
               "Total arguments size: 3012, limit: 2048\"\n");
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(ptx);
        return 2;
    }
    CHECK(cuMemAlloc(&d_out, NOUT * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_out, out, NOUT * sizeof(uint32_t)));

    // The blob is passed BY VALUE: the parameter array holds a pointer to the
    // bytes, exactly as CUDA takes a struct argument.
    void* args[3] = { &d_out, blob, &(unsigned int){N} };
    r = cuLaunchKernel(kernel, 1, 1, 1, NTHREADS, 1, 1, 0, 0, args, NULL);
    if (r != CUDA_SUCCESS) {
        printf("BIGARG TEST: kernel launch failed (expected before the parameter "
               "buffer exists): %d\n", (int)r);
        print_result("launch", r);
        cuMemFree(d_out);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(ptx);
        return 2;
    }

    r = cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        printf("BIGARG TEST: synchronize failed: %d\n", (int)r);
        print_result("synchronize", r);
        cuMemFree(d_out);
        cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(ptx);
        return 2;
    }

    CHECK(cuMemcpyDtoH(out, d_out, NOUT * sizeof(uint32_t)));

    int wrong = 0;
    uint32_t first_got = 0, first_want = 0;
    for (int tid = 0; tid < NOUT; tid++) {
        uint32_t want = (tid < N) ? (base + (uint32_t)(tid % NWORDS)) : 0u;
        if (out[tid] != want) {
            if (!wrong) { first_got = out[tid]; first_want = want; }
            wrong++;
        }
    }

    CHECK(cuMemFree(d_out));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(ptx);

    if (wrong) {
        printf("BIGARG FAILED: %d/%d wrong, first got 0x%08x want 0x%08x\n",
               wrong, NOUT, first_got, first_want);
        return 1;
    }
    printf("BIGARG TEST: %d/%d words correct\n", NOUT, NOUT);
    return 0;
}

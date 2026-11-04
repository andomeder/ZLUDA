#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)

static CUmodule load_mod(CUcontext ctx, const char* ptxname) {
    CUmodule mod;
    FILE* f = fopen(ptxname, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ptxname); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1); fread(ptx, 1, sz, f); ptx[sz] = 0; fclose(f);
    CHECK(cuModuleLoadData(&mod, ptx));
    free(ptx);
    return mod;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    CUdevice dev; CUcontext ctx;
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));

    /* ---- Test 1: global atomics ---- */
    {
        CUmodule mod = load_mod(ctx, "atom_test.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "atom_test"));
        CUdeviceptr dout; CHECK(cuMemAlloc(&dout, sizeof(unsigned)));
        unsigned zero = 0; CHECK(cuMemcpyHtoD(dout, &zero, sizeof(unsigned)));
        /* launch 256 threads, each does atom.add 1 -> expect 256 */
        void* args[1] = { &dout };
        CHECK(cuLaunchKernel(k, 1,1,1, 256,1,1, 0, 0, args, NULL));
        CHECK(cuCtxSynchronize());
        unsigned out = 0; CHECK(cuMemcpyDtoH(&out, dout, sizeof(unsigned)));
        printf("ATOM TEST: 256 threads -> out=%u %s\n", out, out==256?"OK":"FAIL");
        if (out != 256) return 1;
    }

    /* ---- Test 2: dynamic shared memory ---- */
    {
        CUmodule mod = load_mod(ctx, "shmem_test.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "shmem_test"));
        CUdeviceptr dout; CHECK(cuMemAlloc(&dout, sizeof(unsigned)));
        unsigned zero = 0; CHECK(cuMemcpyHtoD(dout, &zero, sizeof(unsigned)));
        /* block of 128 threads: thread i writes i to shared[i], thread 0
           reads back shared[127] -> expect 127. sharedMemBytes = 128*4 */
        void* args[1] = { &dout };
        CHECK(cuLaunchKernel(k, 1,1,1, 128,1,1, 128*4, 0, args, NULL));
        CHECK(cuCtxSynchronize());
        unsigned out = 0; CHECK(cuMemcpyDtoH(&out, dout, sizeof(unsigned)));
        printf("SHMEM TEST: block 128 -> out=%u %s\n", out, out==127?"OK":"FAIL");
        if (out != 127) return 1;
    }

    /* ---- Test 3: shared-memory atomics (classic block reduction) ---- */
    {
        CUmodule mod = load_mod(ctx, "shmem_atom_test.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "shmem_atom_test"));
        CUdeviceptr dout; CHECK(cuMemAlloc(&dout, sizeof(unsigned)));
        unsigned zero = 0; CHECK(cuMemcpyHtoD(dout, &zero, sizeof(unsigned)));
        /* 1 block, 256 threads: each does atom.shared.add to shared[0],
           thread 0 reads it back after barrier -> expect 256 */
        void* args[1] = { &dout };
        CHECK(cuLaunchKernel(k, 1,1,1, 256,1,1, 4, 0, args, NULL));
        CHECK(cuCtxSynchronize());
        unsigned out = 0; CHECK(cuMemcpyDtoH(&out, dout, sizeof(unsigned)));
        printf("SHMEM_ATOM TEST: 256 threads -> out=%u %s\n", out, out==256?"OK":"FAIL");
        if (out != 256) return 1;
    }

    /* ---- Test 4: cross-block atomics (atomic counter across grid) ---- */
    {
        CUmodule mod = load_mod(ctx, "atom_test.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "atom_test"));
        CUdeviceptr dout; CHECK(cuMemAlloc(&dout, sizeof(unsigned)));
        unsigned zero = 0; CHECK(cuMemcpyHtoD(dout, &zero, sizeof(unsigned)));
        /* grid 4x2 = 8 blocks, 32 threads each -> 256 increments */
        void* args[1] = { &dout };
        CHECK(cuLaunchKernel(k, 4,2,1, 32,1,1, 0, 0, args, NULL));
        CHECK(cuCtxSynchronize());
        unsigned out = 0; CHECK(cuMemcpyDtoH(&out, dout, sizeof(unsigned)));
        printf("GRID ATOM TEST: 8 blocks x 32 -> out=%u %s\n", out, out==256?"OK":"FAIL");
        if (out != 256) return 1;
    }

    CHECK(cuCtxDestroy(ctx));
    printf("ALL ATOMIC/SHMEM TESTS PASSED\n");
    return 0;
}

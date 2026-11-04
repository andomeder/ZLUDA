// identify_tile_ranges_test.c - launch identifyTileRanges from
// rasterizer_impl_app.ptx against synthetic sorted keys and verify the
// produced tile ranges.
//
// identifyTileRanges(int L, uint64_t* keys, uint2* ranges):
//   for each i: currtile = (keys[i]>>32)&0x3FFFFF
//   i==0            -> ranges[currtile].x = 0
//   prev != curr    -> ranges[prev].y = i; ranges[curr].x = i
//   i == L-1        -> ranges[currtile].y = L
//
// Test: L=8, tiles [1,1,2,2,2,5,5,9]
//   expect ranges[1]={0,2} ranges[2]={2,5} ranges[5]={5,7} ranges[9]={7,8}
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

typedef struct { unsigned int x, y; } uint2c;

int main(void) {
    enum { L = 8 };
    const unsigned int tiles[L] = {1, 1, 2, 2, 2, 5, 5, 9};
    const int NRANGE = 16;

    uint64_t keys[L];
    for (int i = 0; i < L; i++) keys[i] = ((uint64_t)tiles[i] << 32) | (uint64_t)i;

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction kernel;
    CUdeviceptr d_keys, d_ranges;
    uint2c ranges[NRANGE];

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));

    FILE* f = fopen("/home/cloudsurfer/projects/intel-zluda/tmp/rasterizer_impl_app.ptx", "rb");
    if (!f) { perror("ptx"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1);
    if (fread(ptx, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 1; }
    ptx[sz] = 0;
    fclose(f);

    CHECK(cuModuleLoadData(&mod, ptx));
    printf("module load OK\n");
    CHECK(cuModuleGetFunction(&kernel, mod, "_Z18identifyTileRangesiPmP5uint2"));

    CHECK(cuMemAlloc(&d_keys, L * sizeof(uint64_t)));
    CHECK(cuMemAlloc(&d_ranges, NRANGE * sizeof(uint2c)));
    memset(ranges, 0, sizeof(ranges));
    CHECK(cuMemcpyHtoD(d_keys, keys, L * sizeof(uint64_t)));
    CHECK(cuMemcpyHtoD(d_ranges, ranges, NRANGE * sizeof(uint2c)));

    void* args[3] = { &(int){L}, &d_keys, &d_ranges };
    CHECK(cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(ranges, d_ranges, NRANGE * sizeof(uint2c)));

    int ok = 1;
    struct { unsigned int tile; unsigned int x, y; } expect[] = {
        {1, 0, 2}, {2, 2, 5}, {5, 5, 7}, {9, 7, 8},
    };
    for (unsigned int e = 0; e < 4; e++) {
        uint2c got = ranges[expect[e].tile];
        printf("ranges[%u] = {%u,%u} expect {%u,%u} %s\n",
               expect[e].tile, got.x, got.y, expect[e].x, expect[e].y,
               (got.x == expect[e].x && got.y == expect[e].y) ? "OK" : "MISMATCH");
        if (got.x != expect[e].x || got.y != expect[e].y) ok = 0;
    }

    CHECK(cuMemFree(d_keys));
    CHECK(cuMemFree(d_ranges));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(ptx);
    printf(ok ? "IDENTIFY TILE RANGES OK\n" : "IDENTIFY TILE RANGES FAILED\n");
    return ok ? 0 : 1;
}

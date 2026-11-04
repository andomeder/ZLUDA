// duplicate_with_keys_test.c - launch duplicateWithKeys from
// rasterizer_impl_app.ptx with synthetic data and verify emitted
// key/value pairs. Also exercises the by-value dim3 kernel parameter.
//
// Setup: grid=(4,4,1) [dim3 by value], BLOCK 16, P=3
//   g0: radii=1,  xy=(8,8),  depth=1.5 -> rect 1x1 @ tile(0,0) -> 1 pair
//   g1: radii=20, xy=(30,30), depth=2.5 -> rect x:0..4 y:0..4    -> 16 pairs
//   g2: radii=0 (invisible)                              -> 0 pairs
// offsets = {1, 17, ...}
// Expected: 17 pairs; g0's at pos0 (tile 0), g1's at pos1..16 ordered
// y-major: tile = y*grid.x + x for y in 0..4, x in 0..4.
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

typedef struct { float x, y; } float2c;
typedef struct { unsigned int x, y, z; } dim3c;

static uint32_t fbits(float f) { union { float f; uint32_t u; } v; v.f = f; return v.u; }

int main(void) {
    enum { P = 3, NBUF = 32 };
    const unsigned int GX = 4, GY = 4;

    float2c xy[P] = {{8.0f, 8.0f}, {30.0f, 30.0f}, {0.0f, 0.0f}};
    float depths[P] = {1.5f, 2.5f, 9.9f};
    int radii[P] = {1, 20, 0};
    uint32_t offsets[P] = {1, 17, 17};
    dim3c grid = {GX, GY, 1};

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction kernel;
    CUdeviceptr d_xy, d_depths, d_offsets, d_keys, d_values, d_radii;
    uint64_t keys[NBUF];
    uint32_t values[NBUF];

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
    CHECK(cuModuleGetFunction(&kernel, mod, "_Z17duplicateWithKeysiPK6float2PKfPKjPmPjPi4dim3"));

    CHECK(cuMemAlloc(&d_xy, P * sizeof(float2c)));
    CHECK(cuMemAlloc(&d_depths, P * sizeof(float)));
    CHECK(cuMemAlloc(&d_offsets, P * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_keys, NBUF * sizeof(uint64_t)));
    CHECK(cuMemAlloc(&d_values, NBUF * sizeof(uint32_t)));
    CHECK(cuMemAlloc(&d_radii, P * sizeof(int)));

    memset(keys, 0, sizeof(keys));
    memset(values, 0, sizeof(values));
    CHECK(cuMemcpyHtoD(d_xy, xy, P * sizeof(float2c)));
    CHECK(cuMemcpyHtoD(d_depths, depths, P * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_offsets, offsets, P * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_keys, keys, NBUF * sizeof(uint64_t)));
    CHECK(cuMemcpyHtoD(d_values, values, NBUF * sizeof(uint32_t)));
    CHECK(cuMemcpyHtoD(d_radii, radii, P * sizeof(int)));

    // Args: P, xy, depths, offsets, keys, values, radii, grid(dim3 by value)
    void* args[8] = { &(int){P}, &d_xy, &d_depths, &d_offsets,
                      &d_keys, &d_values, &d_radii, &grid };
    CHECK(cuLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(keys, d_keys, NBUF * sizeof(uint64_t)));
    CHECK(cuMemcpyDtoH(values, d_values, NBUF * sizeof(uint32_t)));

    int ok = 1, n = 0;
    // g0: one pair, tile 0, depth bits of 1.5
    if (keys[0] != (((uint64_t)0 << 32) | fbits(1.5f)) || values[0] != 0) {
        printf("pair[0] key=%llx val=%u MISMATCH\n", (unsigned long long)keys[0], values[0]);
        ok = 0;
    }
    n = 1;
    // g1: 16 pairs at pos 1..16, y-major over 4x4 tiles, depth 2.5
    for (unsigned int y = 0; y < GY && ok; y++) {
        for (unsigned int x = 0; x < GX; x++) {
            uint64_t want_key = ((uint64_t)(y * GX + x) << 32) | fbits(2.5f);
            if (keys[n] != want_key || values[n] != 1) {
                printf("pair[%d] key=%llx val=%u expect tile %u MISMATCH\n",
                       n, (unsigned long long)keys[n], values[n], y * GX + x);
                ok = 0;
                break;
            }
            n++;
        }
    }
    for (int i = n; i < NBUF; i++) {
        if (keys[i] || values[i]) { printf("buffer[%d] not clean\n", i); ok = 0; }
    }
    printf("pairs verified: %d\n", n);

    CHECK(cuMemFree(d_xy)); CHECK(cuMemFree(d_depths)); CHECK(cuMemFree(d_offsets));
    CHECK(cuMemFree(d_keys)); CHECK(cuMemFree(d_values)); CHECK(cuMemFree(d_radii));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(ptx);
    printf(ok ? "DUPLICATE WITH KEYS OK\n" : "DUPLICATE WITH KEYS FAILED\n");
    return ok ? 0 : 1;
}

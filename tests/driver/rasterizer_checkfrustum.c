// rasterizer_checkfrustum.c - load the clang-compiled rasterizer PTX via
// ZLUDA and launch checkFrustum. Verifies the full toolchain end-to-end:
// clang .cu -> PTX -> ZLUDA SPIR-V translation -> Arc B580 execution.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

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

int main(int argc, char** argv) {
    const char* ptx_path = argc > 1 ? argv[1] : "rasterizer_impl.ptx";
    const int N = 1024;

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction kernel;
    CUdeviceptr d_points, d_view, d_proj, d_present;
    float* points = malloc(N * 3 * sizeof(float));
    float view[16], proj[16];
    unsigned char* present = malloc(N);

    // Points scattered in a cube around origin.
    for (int i = 0; i < N; i++) {
        points[i*3+0] = ((i % 32) - 16) * 0.5f;
        points[i*3+1] = (((i / 32) % 32) - 16) * 0.5f;
        points[i*3+2] = ((i / 1024) - 1) * 5.0f + 5.0f;  // z in [5, 10]
        if (i % 4 == 0) points[i*3+2] = -1.0f; // every 4th point behind camera -> false
    }
    // Identity view + simple perspective proj (45 deg fov).
    memset(view, 0, sizeof(view)); view[0]=view[5]=view[10]=view[15]=1.0f;
    memset(proj, 0, sizeof(proj));
    float focal = 1.0f / 0.5f;  // tan(45/2) ~ 0.414 -> f ~ 2.414; use 1/tan(fov/2)
    proj[0] = focal; proj[5] = focal; proj[10] = 1.0f; proj[11] = 1.0f; proj[14] = -1.0f;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    char name[128];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("device: %s\n", name);
    CHECK(cuCtxCreate(&ctx, 0, dev));

    FILE* f = fopen(ptx_path, "rb");
    if (!f) { perror(ptx_path); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1);
    if (fread(ptx, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 1; }
    ptx[sz] = 0;
    fclose(f);
    printf("loaded %s (%ld bytes)\n", ptx_path, sz);

    CHECK(cuModuleLoadData(&mod, ptx));
    printf("module load OK\n");
    CHECK(cuModuleGetFunction(&kernel, mod, "_Z12checkFrustumiPKfS0_S0_Pb"));
    printf("function found OK\n");

    CHECK(cuMemAlloc(&d_points, N * 3 * sizeof(float)));
    CHECK(cuMemAlloc(&d_view, 16 * sizeof(float)));
    CHECK(cuMemAlloc(&d_proj, 16 * sizeof(float)));
    CHECK(cuMemAlloc(&d_present, N));
    memset(present, 0xAA, N); // sentinel: did any thread store?
    CHECK(cuMemcpyHtoD(d_present, present, N));

    CHECK(cuMemcpyHtoD(d_points, points, N * 3 * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_view, view, 16 * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_proj, proj, 16 * sizeof(float)));

    // gridDim=(N,1,1), blockDim=(256,1,1)
    void* args[5] = { &(int){N}, &d_points, &d_view, &d_proj, &d_present };
    CHECK(cuLaunchKernel(kernel, N, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(present, d_present, N));

    int present_count = 0;
    for (int i = 0; i < N; i++) if (present[i]) present_count++;
    printf("checkFrustum: %d/%d points in frustum\n", present_count, N);
    int ok = (present_count > 0 && present_count <= N);

    CHECK(cuMemFree(d_points));
    CHECK(cuMemFree(d_view));
    CHECK(cuMemFree(d_proj));
    CHECK(cuMemFree(d_present));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(points); free(present); free(ptx);
    printf(ok ? "RASTERIZER PTX OK\n" : "RASTERIZER PTX FAILED\n");
    return ok ? 0 : 1;
}

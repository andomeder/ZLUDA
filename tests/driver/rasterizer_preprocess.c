// rasterizer_preprocess.c - load the clang-compiled forward.ptx via ZLUDA and
// launch preprocessCUDA<3>. Verifies the full toolchain end-to-end:
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

#define KERNEL "_Z14preprocessCUDAILi3EEviiiPKfPKN3glm3vecILi3EfLNS2_9qualifierE0EEEfPKNS3_ILi4EfLS4_0EEES1_S1_PbS1_S1_S1_S1_S7_iiffffPiP6float2PfSF_SF_P6float44dim3Pjb"

int main(int argc, char** argv) {
    const char* ptx_path = argc > 1 ? argv[1] : "forward.ptx";
    const int P = 1000;   // gaussians
    const int D = 1;      // SH degree
    const int M = 16;     // max SH coeffs
    const int W = 128, H = 128;
    const float focal = 100.0f, tan_fov = 0.5f;

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction kernel;
    CUdeviceptr d_points, d_scales, d_rot, d_opac, d_shs, d_clamped;
    CUdeviceptr d_view, d_proj, d_cam, d_radii, d_xy, d_depths, d_cov3d, d_colors, d_conic, d_tiles;
    float* points = malloc(P*3*sizeof(float));
    float* scales = malloc(P*3*sizeof(float));
    float* rot = malloc(P*4*sizeof(float));
    float* opac = malloc(P*sizeof(float));
    float* shs = malloc(P*M*3*sizeof(float));
    float view[16], proj[16], cam[3];

    // Populate: gaussian points on a grid in front of camera.
    for (int i = 0; i < P; i++) {
        points[i*3+0] = ((i % 20) - 10) * 0.2f;
        points[i*3+1] = (((i/20) % 20) - 10) * 0.2f;
        points[i*3+2] = 3.0f + (i/400) * 0.5f;
        scales[i*3+0] = scales[i*3+1] = scales[i*3+2] = 0.1f;
        rot[i*4+0] = 1.0f; rot[i*4+1] = rot[i*4+2] = rot[i*4+3] = 0.0f;
        opac[i] = 0.8f;
        shs[i*M*3+0] = 1.0f;  // base color red-ish
    }
    memset(view, 0, sizeof(view)); view[0]=view[5]=view[10]=view[15]=1.0f;
    memset(proj, 0, sizeof(proj));
    proj[0] = focal; proj[5] = focal; proj[10] = 1.0f; proj[11] = 1.0f; proj[14] = -1.0f;
    cam[0]=cam[1]=cam[2]=0.0f;

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
    CHECK(cuModuleGetFunction(&kernel, mod, KERNEL));
    printf("function found OK\n");

    CHECK(cuMemAlloc(&d_points, P*3*sizeof(float)));
    CHECK(cuMemAlloc(&d_scales, P*3*sizeof(float)));
    CHECK(cuMemAlloc(&d_rot, P*4*sizeof(float)));
    CHECK(cuMemAlloc(&d_opac, P*sizeof(float)));
    CHECK(cuMemAlloc(&d_shs, P*M*3*sizeof(float)));
    CHECK(cuMemAlloc(&d_clamped, P*3));
    CHECK(cuMemAlloc(&d_view, 16*sizeof(float)));
    CHECK(cuMemAlloc(&d_proj, 16*sizeof(float)));
    CHECK(cuMemAlloc(&d_cam, 3*sizeof(float)));
    CHECK(cuMemAlloc(&d_radii, P*sizeof(int)));
    CHECK(cuMemAlloc(&d_xy, P*2*sizeof(float)));
    CHECK(cuMemAlloc(&d_depths, P*sizeof(float)));
    CHECK(cuMemAlloc(&d_cov3d, P*6*sizeof(float)));
    CHECK(cuMemAlloc(&d_colors, P*3*sizeof(float)));
    CHECK(cuMemAlloc(&d_conic, P*4*sizeof(float)));
    CHECK(cuMemAlloc(&d_tiles, P*sizeof(unsigned int)));

    CHECK(cuMemcpyHtoD(d_points, points, P*3*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_scales, scales, P*3*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_rot, rot, P*4*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_opac, opac, P*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_shs, shs, P*M*3*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_view, view, 16*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_proj, proj, 16*sizeof(float)));
    CHECK(cuMemcpyHtoD(d_cam, cam, 3*sizeof(float)));

    // grid=(ceil(P/256),1,1), block=(256,1,1); dim3 grid param = (P+255)/256,1,1
    unsigned int grid_dim = (P + 255) / 256;
    void* args[30] = {
        &(int){P}, &(int){D}, &(int){M},
        &d_points, &d_scales, &(float){1.0f}, &d_rot, &d_opac, &d_shs,
        &d_clamped, &(CUdeviceptr){0}, &(CUdeviceptr){0},   // cov3D_precomp, colors_precomp = NULL
        &d_view, &d_proj, &d_cam,
        &(int){W}, &(int){H},
        &(float){focal}, &(float){focal}, &(float){tan_fov}, &(float){tan_fov},
        &d_radii, &d_xy, &d_depths, &d_cov3d, &d_colors, &d_conic,
        &(unsigned int){grid_dim}, &d_tiles, &(bool){0}
    };
    CHECK(cuLaunchKernel(kernel, grid_dim, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    printf("kernel launched OK\n");

    // Read back radii - should be >= 0 and mostly small (in-frustum points)
    int* radii = malloc(P*sizeof(int));
    float* depths = malloc(P*sizeof(float));
    CHECK(cuMemcpyDtoH(radii, d_radii, P*sizeof(int)));
    CHECK(cuMemcpyDtoH(depths, d_depths, P*sizeof(float)));
    int nz = 0; float maxd = 0;
    for (int i = 0; i < P; i++) { if (radii[i] > 0) nz++; if (depths[i] > maxd) maxd = depths[i]; }
    printf("radii>0: %d/%d, max depth: %.3f\n", nz, P, maxd);
    int ok = (nz > 0);

    CHECK(cuMemFree(d_points)); CHECK(cuMemFree(d_scales)); CHECK(cuMemFree(d_rot));
    CHECK(cuMemFree(d_opac)); CHECK(cuMemFree(d_shs)); CHECK(cuMemFree(d_clamped));
    CHECK(cuMemFree(d_view)); CHECK(cuMemFree(d_proj)); CHECK(cuMemFree(d_cam));
    CHECK(cuMemFree(d_radii)); CHECK(cuMemFree(d_xy)); CHECK(cuMemFree(d_depths));
    CHECK(cuMemFree(d_cov3d)); CHECK(cuMemFree(d_colors)); CHECK(cuMemFree(d_conic));
    CHECK(cuMemFree(d_tiles));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(points); free(scales); free(rot); free(opac); free(shs); free(ptx); free(radii); free(depths);
    printf(ok ? "PREPROCESS KERNEL OK\n" : "PREPROCESS KERNEL FAILED\n");
    return ok ? 0 : 1;
}

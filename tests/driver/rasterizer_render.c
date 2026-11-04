// rasterizer_render.c - launch renderCUDA<3> directly via the driver API.
// Verifies the render path (shared-mem + CUB binning kernels) on ZLUDA/B580.
// renderCUDA signature (from forward.cu):
//   (uint2* ranges, uint* point_list, int W, int H, float2* means2D,
//    float* colors, float4* conic_opacity, float* final_T, uint* n_contrib,
//    float* bg_color, float* out_color)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); fprintf(stderr, "FAIL at %s:%d: %s (%s)\n", __FILE__, __LINE__, #x, s ? s : "?"); return 1; } } while (0)

#define KERNEL "_Z10renderCUDAILj3EEvPK5uint2PKjiiPK6float2PKfPK6float4PfPjS9_SD_"

int main(int argc, char** argv) {
    const char* ptx_path = argc > 1 ? argv[1] : "forward.ptx";
    const int W = 128, H = 128;
    const int NTILES = ((W + 15) / 16) * ((H + 15) / 16);   // 8x8 = 64 tiles
    const int P = 1000;

    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction kernel;
    CUdeviceptr d_ranges, d_point_list, d_means2D, d_colors, d_conic, d_finalT, d_ncontrib, d_bg, d_out;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
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

    // ranges: one per tile, point_list dense (assume preprocess already ran).
    CHECK(cuMemAlloc(&d_ranges, NTILES * 2 * sizeof(unsigned int)));
    CHECK(cuMemAlloc(&d_point_list, P * sizeof(unsigned int)));
    CHECK(cuMemAlloc(&d_means2D, P * 2 * sizeof(float)));
    CHECK(cuMemAlloc(&d_colors, P * 3 * sizeof(float)));
    CHECK(cuMemAlloc(&d_conic, P * 4 * sizeof(float)));
    CHECK(cuMemAlloc(&d_finalT, W * H * sizeof(float)));
    CHECK(cuMemAlloc(&d_ncontrib, W * H * sizeof(unsigned int)));
    CHECK(cuMemAlloc(&d_bg, 3 * sizeof(float)));
    CHECK(cuMemAlloc(&d_out, 3 * W * H * sizeof(float)));

    unsigned int* ranges = malloc(NTILES * 2 * sizeof(unsigned int));
    unsigned int* plist = malloc(P * sizeof(unsigned int));
    float* means2D = malloc(P * 2 * sizeof(float));
    float* colors = malloc(P * 3 * sizeof(float));
    float* conic = malloc(P * 4 * sizeof(float));
    float* bg = malloc(3 * sizeof(float));
    for (int t = 0; t < NTILES; t++) { ranges[2*t] = 0; ranges[2*t+1] = 0; }
    // Sprinkle points across the image so some tiles have work.
    for (int i = 0; i < P; i++) {
        plist[i] = i;
        means2D[2*i] = (float)((i * 37) % W);
        means2D[2*i+1] = (float)((i * 53) % H);
        colors[3*i] = 1.0f; colors[3*i+1] = 0.5f; colors[3*i+2] = 0.25f;
        conic[4*i] = 1.0f; conic[4*i+1] = 0.0f; conic[4*i+2] = 1.0f; conic[4*i+3] = 0.9f;
        // assign to tile (i*37 % W)/16, (i*53 % H)/16
        int tx = ((int)means2D[2*i]) / 16, ty = ((int)means2D[2*i+1]) / 16;
        int tile = ty * (W/16) + tx;
        if (tile >= 0 && tile < NTILES) { ranges[2*tile] = i; ranges[2*tile+1] = i + 1; }
    }
    bg[0] = bg[1] = bg[2] = 0.0f;

    CHECK(cuMemcpyHtoD(d_ranges, ranges, NTILES * 2 * sizeof(unsigned int)));
    CHECK(cuMemcpyHtoD(d_point_list, plist, P * sizeof(unsigned int)));
    CHECK(cuMemcpyHtoD(d_means2D, means2D, P * 2 * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_colors, colors, P * 3 * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_conic, conic, P * 4 * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_bg, bg, 3 * sizeof(float)));

    void* args[11] = {
        &d_ranges, &d_point_list,
        &(int){W}, &(int){H},
        &d_means2D, &d_colors, &d_conic,
        &d_finalT, &d_ncontrib, &d_bg, &d_out
    };
    CHECK(cuLaunchKernel(kernel, NTILES, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    printf("kernel launched OK\n");

    float* out = malloc(3 * W * H * sizeof(float));
    CHECK(cuMemcpyDtoH(out, d_out, 3 * W * H * sizeof(float)));
    float maxv = 0; int nz = 0;
    for (int i = 0; i < 3 * W * H; i++) { if (out[i] > 0.001f) nz++; if (out[i] > maxv) maxv = out[i]; }
    printf("out_color nonzero pixels: %d, max value: %.3f\n", nz, maxv);
    int ok = (nz > 0 && maxv > 0.1f);

    cuMemFree(d_ranges); cuMemFree(d_point_list); cuMemFree(d_means2D);
    cuMemFree(d_colors); cuMemFree(d_conic); cuMemFree(d_finalT);
    cuMemFree(d_ncontrib); cuMemFree(d_bg); cuMemFree(d_out);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    free(ptx); free(ranges); free(plist); free(means2D); free(colors);
    free(conic); free(bg); free(out);
    printf(ok ? "RENDER KERNEL OK\n" : "RENDER KERNEL FAILED\n");
    return ok ? 0 : 1;
}

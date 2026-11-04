// memcpy2d3d_test.c: verify cuMemcpy2D_v2 / cuMemcpy3D_v2 (strided copies)
// between device buffers with pitch, and host<->device 2D copies.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                             \
    do {                                                                        \
        CUresult err = (call);                                                  \
        if (err != CUDA_SUCCESS) {                                              \
            fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__,      \
                    #call, err);                                                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define W 16   /* elements per row */
#define H 8    /* rows */
#define DEPTH 4

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr d_src = 0, d_dst = 0;
    CUdeviceptr d_pitch = 0;
    size_t pitch = 0;
    int *h_src, *h_dst;
    int x, y, z, i;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));

    h_src = malloc(W * H * sizeof(int));
    h_dst = malloc(W * H * sizeof(int));
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            h_src[y * W + x] = y * 100 + x;

    CHECK(cuMemAlloc(&d_src, W * H * sizeof(int)));
    CHECK(cuMemAlloc(&d_dst, W * H * sizeof(int)));
    CHECK(cuMemAllocPitch(&d_pitch, &pitch, W * sizeof(int), H, sizeof(int)));

    // --- 2D device->device with pitch (rows may be padded) ---
    CUDA_MEMCPY2D c2d;
    memset(&c2d, 0, sizeof(c2d));
    c2d.srcMemoryType = CU_MEMORYTYPE_HOST;
    c2d.srcHost = h_src;
    c2d.srcPitch = W * sizeof(int);
    c2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    c2d.dstDevice = d_pitch;
    c2d.dstPitch = pitch;
    c2d.WidthInBytes = W * sizeof(int);
    c2d.Height = H;
    CHECK(cuMemcpy2D(&c2d));

    // Copy back via 2D and verify
    memset(h_dst, 0, W * H * sizeof(int));
    CUDA_MEMCPY2D c2d_back;
    memset(&c2d_back, 0, sizeof(c2d_back));
    c2d_back.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    c2d_back.srcDevice = d_pitch;
    c2d_back.srcPitch = pitch;
    c2d_back.dstMemoryType = CU_MEMORYTYPE_HOST;
    c2d_back.dstHost = h_dst;
    c2d_back.dstPitch = W * sizeof(int);
    c2d_back.WidthInBytes = W * sizeof(int);
    c2d_back.Height = H;
    CHECK(cuMemcpy2D(&c2d_back));

    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            if (h_dst[y * W + x] != h_src[y * W + x]) {
                fprintf(stderr, "FAIL: 2D [%d,%d] %d != %d\n", y, x,
                        h_dst[y * W + x], h_src[y * W + x]);
                exit(1);
            }

    // --- 3D device->device copy ---
    // Build a 3D block in d_src, copy to d_dst with cuMemcpy3D.
    CHECK(cuMemcpyHtoD(d_src, h_src, W * H * sizeof(int)));

    memset(h_dst, 0, W * H * sizeof(int));
    CUDA_MEMCPY3D c3d;
    memset(&c3d, 0, sizeof(c3d));
    c3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    c3d.srcDevice = d_src;
    c3d.srcPitch = W * sizeof(int);
    c3d.srcHeight = H / 2;
    c3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    c3d.dstDevice = d_dst;
    c3d.dstPitch = W * sizeof(int);
    c3d.dstHeight = H / 2;
    c3d.WidthInBytes = W * sizeof(int);
    c3d.Height = H / 2;
    c3d.Depth = 2;
    CHECK(cuMemcpy3D(&c3d));

    CHECK(cuMemcpyDtoH(h_dst, d_dst, W * H * sizeof(int)));
    for (i = 0; i < W * H; i++)
        if (h_dst[i] != h_src[i]) {
            fprintf(stderr, "FAIL: 3D [%d] %d != %d\n", i, h_dst[i], h_src[i]);
            exit(1);
        }

    // Verify all copied elements (2 slices x H/2 rows each)
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            if (h_dst[y * W + x] != h_src[y * W + x]) {
                fprintf(stderr, "FAIL: 3D depth [%d,%d] %d != %d\n", y, x,
                        h_dst[y * W + x], h_src[y * W + x]);
                exit(1);
            }

    CHECK(cuMemFree(d_src));
    CHECK(cuMemFree(d_dst));
    CHECK(cuMemFree(d_pitch));
    CHECK(cuCtxDestroy(ctx));
    free(h_src);
    free(h_dst);

    printf("ALL 2D/3D MEMCPY TESTS PASSED (pitch=%zu)\n", pitch);
    return 0;
}

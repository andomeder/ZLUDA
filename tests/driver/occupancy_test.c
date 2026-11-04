// occupancy_test.c: the cuOccupancy* queries an application uses to pick a
// launch configuration (llama.cpp asks for the flash attention block count,
// torch for its elementwise kernels). The numbers have to respect the
// device limits: threads and shared memory per multiprocessor, and the cap
// on resident blocks.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static const char *PTX =
    ".version 6.5\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry k_small(.param .u64 pa)\n"
    "{\n"
    "  .reg .b64 %rd1;\n"
    "  ld.param.u64 %rd1, [pa];\n"
    "  ret;\n"
    "}\n";

static size_t b2d(int block_size) { return (size_t)block_size * 64; }

int main(void) {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    int v = 0;
    if (cuInit(0) != CUDA_SUCCESS) return 1;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 1;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) return 1;
    if (cuModuleLoadData(&mod, PTX) != CUDA_SUCCESS) { printf("load failed\n"); return 1; }
    if (cuModuleGetFunction(&fn, mod, "k_small") != CUDA_SUCCESS) { printf("fn failed\n"); return 1; }

    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    printf("MAX_THREADS_PER_MP      = %d\n", v);
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, dev);
    printf("MAX_BLOCKS_PER_MP       = %d\n", v);
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
    printf("MAX_SHARED_PER_MP       = %d\n", v);
    cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    printf("MULTIPROCESSOR_COUNT    = %d\n", v);

    int sizes[5] = {32, 128, 256, 512, 1024};
    int bad = 0;
    for (int i = 0; i < 5; i++) {
        int blocks = -1;
        CUresult r = cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, sizes[i], 0);
        printf("block=%4d -> %d blocks/sm (rc=%d)\n", sizes[i], blocks, r);
        if (r != CUDA_SUCCESS || blocks <= 0) bad++;
    }
    /* Dynamic shared memory has to reduce the number of resident blocks. */
    int blocks_no_smem = 0, blocks_big_smem = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_no_smem, fn, 256, 0);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_big_smem, fn, 256, 48 * 1024);
    printf("256 threads: %d blocks with no shared memory, %d with 48 KiB\n",
           blocks_no_smem, blocks_big_smem);
    if (blocks_big_smem > blocks_no_smem) bad++;

    int min_grid = 0, block = 0;
    CUresult r = cuOccupancyMaxPotentialBlockSize(&min_grid, &block, fn, b2d, 0, 0);
    printf("potential block size = %d, min grid = %d (rc=%d)\n", block, min_grid, r);
    if (r != CUDA_SUCCESS || block <= 0 || min_grid <= 0) bad++;

    size_t avail = 0;
    r = cuOccupancyAvailableDynamicSMemPerBlock(&avail, fn, 2, 256);
    printf("available dynamic shared memory per block (2 blocks) = %zu bytes (rc=%d)\n",
           avail, r);
    if (r != CUDA_SUCCESS || avail == 0) bad++;

    printf("OCCUPANCY %s\n", bad ? "FAILED" : "OK");
    return bad != 0;
}

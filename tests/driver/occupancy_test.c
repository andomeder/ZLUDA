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

    /* The two caps, checked against what the hardware delivers: a group that
       spins on a host word until it is released (tmp/harness/hangdiag/
       resident_probe.c) measures 320 resident workgroups at 128 threads, 640 at
       64, 1280 at 32, 16 and 8 - i.e. threads-per-multiprocessor binds down to
       128 and the block cap binds below it. The query has to reproduce both,
       because an application picks its grid from these numbers. */
    int threads_per_mp = 0, max_blocks = 0, mps = 0;
    cuDeviceGetAttribute(&threads_per_mp, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&max_blocks, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&mps, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    int occ128 = 0, occ16 = 0, occ32 = 0, occ1024 = 0;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ128, fn, 128, 0);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ16, fn, 16, 0);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ32, fn, 32, 0);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ1024, fn, 1024, 0);
    printf("multiprocessors %d x blocks(128) %d = %d resident workgroups\n", mps, occ128,
           mps * occ128);
    if (occ128 != threads_per_mp / 128) {
        printf("FAIL: 128-thread occupancy should be the thread limit (%d)\n",
               threads_per_mp / 128);
        bad++;
    }
    if (occ1024 != threads_per_mp / 1024) {
        printf("FAIL: 1024-thread occupancy should be the thread limit (%d)\n",
               threads_per_mp / 1024);
        bad++;
    }
    if (occ16 != max_blocks || occ32 != max_blocks) {
        printf("FAIL: a small group should reach the block cap (%d), got %d and %d\n",
               max_blocks, occ16, occ32);
        bad++;
    }
    if (mps <= 1 || mps > 256) {
        printf("FAIL: implausible multiprocessor count %d\n", mps);
        bad++;
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

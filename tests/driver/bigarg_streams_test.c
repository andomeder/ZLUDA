/* Two streams using the same oversized-argument kernel at once.
 *
 * A kernel whose arguments exceed the compiler's 2048 byte limit reads them from
 * a buffer the driver fills per launch, out of a ring of slots. The ring is per
 * stream, because "has the launch that read this slot retired" is answered with
 * the stream's own markers: a ring shared between two streams could be
 * overwritten while the other stream's kernel was still reading it.
 *
 * The kernel below reads a word out of its 4096 byte by-value block and writes it
 * out, so each stream's result must be that stream's own block content. Fifteen
 * hundred launches across two streams, then every result checked. A shared ring
 * shows up as one stream reading the other's data.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WORDS 1024        /* 4096 bytes, over the 2048 limit */
#define OUT_WORDS 64

static const char *PTX =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry pick(.param .u64 out, .param .align 4 .b8 block[4096],\n"
    "                     .param .u32 words)\n"
    "{\n"
    "  .reg .pred %p1;\n"
    "  .reg .b32 %r<8>;\n"
    "  .reg .b64 %rd<8>;\n"
    "  ld.param.u64 %rd1, [out];\n"
    "  ld.param.u32 %r1, [words];\n"
    "  ld.param.u32 %r3, [block+0];\n"
    "  mov.u32 %r2, %tid.x;\n"
    "  setp.ge.u32 %p1, %r2, %r1;\n"
    "  @%p1 bra L_done;\n"
    "  mul.wide.u32 %rd2, %r2, 4;\n"
    "  add.s64 %rd3, %rd1, %rd2;\n"
    "  st.global.u32 [%rd3], %r3;\n"
    "L_done:\n"
    "  ret;\n"
    "}\n";

#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
    const char *n; cuGetErrorName(r, &n); \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, n); return 1; } } while (0)

int main(void) {
    CHECK(cuInit(0));
    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, PTX));
    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, mod, "pick"));

    CUstream streams[2];
    CHECK(cuStreamCreate(&streams[0], 0));
    CHECK(cuStreamCreate(&streams[1], 0));

    CUdeviceptr out[2];
    for (int s = 0; s < 2; s++) {
        CHECK(cuMemAlloc(&out[s], OUT_WORDS * 4));
    }

    /* Each stream carries its own marker value in its argument block. */
    unsigned int *blocks[2];
    const unsigned int marker[2] = {0xA5A50001u, 0x5A5A0002u};
    for (int s = 0; s < 2; s++) {
        blocks[s] = malloc(BLOCK_WORDS * 4);
        for (int i = 0; i < BLOCK_WORDS; i++) {
            blocks[s][i] = marker[s];
        }
    }
    unsigned words = OUT_WORDS;

    const int rounds = 750;
    int wrong = 0;
    for (int round = 0; round < rounds; round++) {
        for (int s = 0; s < 2; s++) {
            void *args[3] = {&out[s], blocks[s], &words};
            CHECK(cuLaunchKernel(kernel, 1, 1, 1, OUT_WORDS, 1, 1, 0, streams[s], args, NULL));
        }
    }
    for (int s = 0; s < 2; s++) {
        CHECK(cuStreamSynchronize(streams[s]));
        unsigned int *got = malloc(OUT_WORDS * 4);
        CHECK(cuMemcpyDtoH(got, out[s], OUT_WORDS * 4));
        for (int i = 0; i < OUT_WORDS; i++) {
            if (got[i] != marker[s]) {
                wrong++;
            }
        }
        free(got);
    }

    printf("BIGARG STREAMS TEST: %d launches across two streams, %d/%d words wrong\n",
           2 * rounds, wrong, OUT_WORDS * 2);
    for (int s = 0; s < 2; s++) {
        free(blocks[s]);
        CHECK(cuStreamDestroy(streams[s]));
        CHECK(cuMemFree(out[s]));
    }
    if (wrong) {
        printf("BIGARG STREAMS TEST: FAILED (a stream read the other's arguments)\n");
        return 1;
    }
    printf("BIGARG STREAMS TEST: OK\n");
    return 0;
}

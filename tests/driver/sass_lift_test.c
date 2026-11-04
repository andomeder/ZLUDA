/* A fatbin that carries ONLY compiled code (NVIDIA SASS) has to be refused
 * without the SASS lift and has to run with it.
 *
 * The image is a real kernel taken out of libtorch_cuda.so: the 4-wide
 * vectorized elementwise multiply of TensorIterator, whose second argument is
 * a by-value std::array<char*, 3>.  The record is kind=2 (ELF/cubin), which is
 * the shape `-gencode arch=compute_75,code=sm_75` produces and the one our
 * loader refuses today.
 *
 * Usage:
 *   ./sass_lift_test [container.fatbin]
 *
 *   without ZLUDA_SASS_LIFT   the load must fail with CUDA_ERROR_NOT_SUPPORTED
 *   with    ZLUDA_SASS_LIFT=1 the kernel is lifted, built and must compute
 *                             out[i] = in1[i] * in2[i] correctly
 *
 * Regenerate the container from a cubin with
 *   python3 mkfatbin.py <kernel.cubin> sass_mul --elf
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#define CHECK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    const char* name_; const char* str_; \
    cuGetErrorName(r_, &name_); cuGetErrorString(r_, &str_); \
    fprintf(stderr, "FAIL %s:%d %s %s\n", __FILE__, __LINE__, name_, str_); \
    exit(1); } } while (0)

/* _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIfffNS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_ */
#define MUL_KERNEL \
    "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13BinaryFunctorIfffNS0_15" \
    "binary_internal10MulFunctorIfEEEESt5arrayIPcLm3EEEEviT0_T1_"

#define BLOCK 128   /* the kernel's .maxntid */
#define VEC   4     /* vectorized_elementwise_kernel<4, ...> */
#define PER_BLOCK (BLOCK * VEC)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void* load_file(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

/* The parameter block the kernel reads from the constant bank:
 *   +0x160 int N        +0x164 func_t (1 byte)      +0x168 char* data[3]
 * which is what the CUDA runtime passes for
 *   (int N, BinaryFunctor<...>, std::array<char*, 3>).
 */
struct mul_args {
    int n;
    unsigned char func;
    char* data[3];
};

static int run_kernel(CUfunction kernel, int n,
                      CUdeviceptr out, CUdeviceptr in1, CUdeviceptr in2) {
    struct mul_args args;
    args.n = n;
    args.func = 0;
    args.data[0] = (char*)out;
    args.data[1] = (char*)in1;
    args.data[2] = (char*)in2;
    void* params[3] = { &args.n, &args.func, &args.data };
    unsigned grid = (unsigned)((n + PER_BLOCK - 1) / PER_BLOCK);
    CUresult r = cuLaunchKernel(kernel, grid, 1, 1, BLOCK, 1, 1, 0, 0, params, NULL);
    if (r != CUDA_SUCCESS) {
        const char* name_; const char* str_;
        cuGetErrorName(r, &name_); cuGetErrorString(r, &str_);
        printf("  launch failed: %s %s\n", name_, str_);
        return -1;
    }
    CHECK(cuCtxSynchronize());
    return 0;
}

/* Fill, run and compare against the CPU. Returns the number of wrong
 * elements, or -1 when the launch failed. */
static int verify(CUfunction kernel, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float* h_in1 = malloc(bytes);
    float* h_in2 = malloc(bytes);
    float* h_out = malloc(bytes);
    float* h_ref = malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_in1[i] = (float)((i % 97) * 0.5f + 0.25f);
        h_in2[i] = (float)(2.0f - (i % 89) * 0.125f);
        h_ref[i] = h_in1[i] * h_in2[i];
        h_out[i] = -12345.0f;
    }
    CUdeviceptr d_out, d_in1, d_in2;
    CHECK(cuMemAlloc(&d_out, bytes));
    CHECK(cuMemAlloc(&d_in1, bytes));
    CHECK(cuMemAlloc(&d_in2, bytes));
    CHECK(cuMemcpyHtoD(d_in1, h_in1, bytes));
    CHECK(cuMemcpyHtoD(d_in2, h_in2, bytes));
    {   /* poison the output so a kernel that does nothing cannot pass */
        uint32_t poison = 0xDEADBEEFu;
        CHECK(cuMemsetD32(d_out, poison, (size_t)n));
    }
    if (run_kernel(kernel, n, d_out, d_in1, d_in2) != 0) {
        cuMemFree(d_out); cuMemFree(d_in1); cuMemFree(d_in2);
        free(h_in1); free(h_in2); free(h_out); free(h_ref);
        return -1;
    }
    CHECK(cuMemcpyDtoH(h_out, d_out, bytes));
    int wrong = 0;
    for (int i = 0; i < n; i++) {
        uint32_t a, b;
        memcpy(&a, &h_out[i], 4);
        memcpy(&b, &h_ref[i], 4);
        if (a != b) {
            if (wrong < 3) {
                printf("    element %d: gpu %g cpu %g\n", i, h_out[i], h_ref[i]);
            }
            wrong++;
        }
    }
    cuMemFree(d_out); cuMemFree(d_in1); cuMemFree(d_in2);
    free(h_in1); free(h_in2); free(h_out); free(h_ref);
    return wrong;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "sass_mul.fatbin";
    size_t size = 0;
    void* image = load_file(path, &size);
    const char* lift = getenv("ZLUDA_SASS_LIFT");
    const int expecting_lift = lift != NULL && (strcmp(lift, "1") == 0 || strcmp(lift, "true") == 0);
    printf("container %s (%zu bytes), ZLUDA_SASS_LIFT %s\n",
           path, size, lift ? lift : "(unset)");

    CUdevice dev;
    CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));

    /* 1: cuLibraryLoadData + cuLibraryGetKernel (the path libcudart uses) */
    {
        double t0 = now_ms();
        CUlibrary lib = NULL;
        CUresult r = cuLibraryLoadData(&lib, image, NULL, NULL, 0, NULL, NULL, 0);
        double load_ms = now_ms() - t0;
        if (r != CUDA_SUCCESS) {
            const char* name_; const char* str_;
            cuGetErrorName(r, &name_); cuGetErrorString(r, &str_);
            printf("cuLibraryLoadData: %s %s (%.1f ms)\n", name_, str_, load_ms);
            if (!expecting_lift && r == CUDA_ERROR_NOT_SUPPORTED) {
                printf("REFUSED WITHOUT THE LIFT: correct, a SASS-only image is not "
                       "translatable on its own\n");
                CHECK(cuCtxDestroy(ctx));
                printf("SASS FATBIN TEST PASSED (refusal path)\n");
                return 0;
            }
            printf("SASS FATBIN TEST FAILED\n");
            return 1;
        }
        printf("cuLibraryLoadData: OK (%.1f ms)\n", load_ms);
        CUkernel kern = NULL;
        double t1 = now_ms();
        CHECK(cuLibraryGetKernel(&kern, lib, MUL_KERNEL));
        double build_ms = now_ms() - t1;
        printf("cuLibraryGetKernel: OK (%.1f ms, includes lift + SPIR-V + IGC)\n", build_ms);
        int wrong = verify((CUfunction)kern, 4096);
        if (wrong < 0) { printf("SASS FATBIN TEST FAILED\n"); return 1; }
        printf("  N=4096 grid=%d: wrong %d/4096 %s\n", 4096 / PER_BLOCK, wrong,
               wrong == 0 ? "OK" : "FAIL");
        if (wrong != 0) { printf("SASS FATBIN TEST FAILED\n"); return 1; }
        wrong = verify((CUfunction)kern, 3000);
        if (wrong < 0) { printf("SASS FATBIN TEST FAILED\n"); return 1; }
        printf("  N=3000 (tail path): wrong %d/3000 %s\n", wrong, wrong == 0 ? "OK" : "FAIL");
        if (wrong != 0) { printf("SASS FATBIN TEST FAILED\n"); return 1; }
    }

    /* 2: cuModuleLoadFatBinary + cuModuleGetFunction (the module path) */
    {
        double t0 = now_ms();
        CUmodule mod = NULL;
        CUresult r = cuModuleLoadFatBinary(&mod, image);
        double load_ms = now_ms() - t0;
        if (r != CUDA_SUCCESS) {
            const char* name_; const char* str_;
            cuGetErrorName(r, &name_); cuGetErrorString(r, &str_);
            printf("cuModuleLoadFatBinary: %s %s (%.1f ms)\n", name_, str_, load_ms);
            printf("SASS FATBIN TEST FAILED\n");
            return 1;
        }
        printf("cuModuleLoadFatBinary: OK (%.1f ms)\n", load_ms);
        CUfunction kern = NULL;
        double t1 = now_ms();
        CHECK(cuModuleGetFunction(&kern, mod, MUL_KERNEL));
        printf("cuModuleGetFunction: OK (%.1f ms)\n", now_ms() - t1);
        int wrong = verify(kern, 4096);
        if (wrong < 0) { printf("SASS FATBIN TEST FAILED\n"); return 1; }
        printf("  N=4096 via the module path: wrong %d/4096 %s\n", wrong,
               wrong == 0 ? "OK" : "FAIL");
        if (wrong != 0) { printf("SASS FATBIN TEST FAILED\n"); return 1; }
    }

    CHECK(cuCtxDestroy(ctx));
    printf("SASS FATBIN TEST PASSED\n");
    return 0;
}

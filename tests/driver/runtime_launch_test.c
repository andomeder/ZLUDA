// runtime_launch_test.c - exercise the CUDA RUNTIME API launch path against
// ZLUDA using real libcudart.so.12.
//
// Compiled nvcc apps register their kernels via __cudaRegisterFatBinary /
// __cudaRegisterFunction (inside libcudart) and launch via cudaLaunchKernel.
// libcudart lazily JITs each module by calling OUR cuModuleLoadDataEx with
// the registered fatbin image. This test replicates that flow by hand:
//
//   1. dlopen real libcudart.so.12
//   2. build a fatbinc wrapper around a packed PTX fatbin (mkfatbin.py)
//   3. __cudaRegisterFatBinary + __cudaRegisterFunction(identifyTileRanges)
//   4. cudaMalloc/cudaMemcpy setup, cudaLaunchKernel, cudaMemcpy back
//   5. verify tile ranges like identify_tile_ranges_test.c
//
// Success proves the runtime-API plumbing: fatbin -> extract_ptx ->
// SPIR-V -> IGC -> L0 execution, driven entirely by stock libcudart.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

// --- minimal cudart type shims (avoid pulling full runtime headers) ---
typedef int cudaError_t_;
typedef struct { unsigned int x, y, z; } dim3s;
typedef struct { unsigned int x, y; } uint2c;

// __fatBinC_Wrapper_t layout (cuda_fatbinary.h): magic, version, data ptr,
// filename_or_fatbins ptr.
struct fatbinc_wrapper {
    int magic;
    int version;
    void *data;
    void *filename_or_fatbins;
};

#define FATBINC_MAGIC 0x466243B1
#define FATBINC_VERSION 1

static const char *CUDART_PATH =
    "/home/cloudsurfer/projects/mvsgaussian-xpu-test/.venv-cuda/lib/python3.10/"
    "site-packages/nvidia/cuda_runtime/lib/libcudart.so.12";
static const char *FATBINC_PATH =
    "/home/cloudsurfer/projects/intel-zluda/ZLUDA/tests/driver/runtime_impl.fatbinc";

typedef void **(*reg_fatbin_fn)(void *);
typedef void (*reg_func_fn)(void **, const void *, char *, const char *, int,
                            uint32_t *, uint32_t *, uint32_t *, uint32_t *,
                            uint32_t **);
typedef int (*launch_fn)(const void *, dim3s, dim3s, void **, size_t, void *);
typedef int (*memget_fn)(void **, size_t);
typedef int (*freer_fn)(void *);
typedef int (*memcpy_fn)(void *, const void *, size_t, int);
typedef const char *(*errstr_fn)(int);

static void die(const char *what, int rc) {
    fprintf(stderr, "FAIL at %s rc=%d\n", what, rc);
    exit(1);
}

// Dummy host-side stub standing in for the compiler-generated host function
// pointer. Only its identity matters; it is never executed.
static void identifyTileRanges_hoststub(void) {}

int main(void) {
    enum { L = 8 };
    const unsigned int tiles[L] = {1, 1, 2, 2, 2, 5, 5, 9};
    enum { NRANGE = 16 };
    int rc;

    void *rt = dlopen(CUDART_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!rt) {
        fprintf(stderr, "dlopen libcudart failed: %s\n", dlerror());
        return 1;
    }
    reg_fatbin_fn p_reg_fatbin =
        (reg_fatbin_fn)dlsym(rt, "__cudaRegisterFatBinary");
    reg_func_fn p_reg_func =
        (reg_func_fn)dlsym(rt, "__cudaRegisterFunction");
    reg_func_fn p_reg_func_v2 =
        (reg_func_fn)dlsym(rt, "__cudaRegisterFunction_v2");
    launch_fn p_launch = (launch_fn)dlsym(rt, "cudaLaunchKernel");
    memget_fn p_malloc = (memget_fn)dlsym(rt, "cudaMalloc");
    freer_fn p_free = (freer_fn)dlsym(rt, "cudaFree");
    memcpy_fn p_memcpy = (memcpy_fn)dlsym(rt, "cudaMemcpy");
    errstr_fn p_errstr = (errstr_fn)dlsym(rt, "cudaGetErrorString");
    if (!p_reg_fatbin || !p_reg_func || !p_launch || !p_malloc || !p_free ||
        !p_memcpy || !p_errstr) {
        fprintf(stderr,
                "dlsym failed: fatbin=%p func=%p launch=%p malloc=%p "
                "free=%p memcpy=%p errstr=%p\n",
                (void *)p_reg_fatbin, (void *)p_reg_func, (void *)p_launch,
                (void *)p_malloc, (void *)p_free, (void *)p_memcpy,
                (void *)p_errstr);
        return 1;
    }
    // v2 variant preferred when present (CUDA >= 12 uses it).
    reg_func_fn reg_func = p_reg_func_v2 ? p_reg_func_v2 : p_reg_func;
    printf("libcudart loaded; symbols resolved (RegisterFunction_v2=%d)\n",
           p_reg_func_v2 != NULL);

    // Read the packed fatbinc blob: [magic u32][version u32][fatbin frame].
    FILE *f = fopen(FATBINC_PATH, "rb");
    if (!f) {
        perror(FATBINC_PATH);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc(sz);
    if (fread(blob, 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "short read\n");
        return 1;
    }
    fclose(f);

    unsigned int magic = *(unsigned int *)blob;
    unsigned int version = *(unsigned int *)(blob + 4);
    printf("fatbinc blob %ld bytes magic=0x%X version=%u\n", sz, magic,
           version);
    if (magic != FATBINC_MAGIC || version != FATBINC_VERSION) {
        fprintf(stderr, "unexpected wrapper header\n");
        return 1;
    }

    // Runtime init (lazy context creation). cudaFree(NULL) is a valid no-op
    // that forces the runtime to initialize against our driver.
    void *dummy = NULL;
    rc = p_free(&dummy);
    if (rc != 0) {
        fprintf(stderr, "cudaFree(0) init failed: %d (%s)\n", rc,
                p_errstr(rc));
        return 1;
    }
    printf("runtime init OK\n");

    // Register the fatbin + the kernel entry.
    struct fatbinc_wrapper w;
    w.magic = FATBINC_MAGIC;
    w.version = FATBINC_VERSION;
    w.data = blob + 8; // inline FatbinHeader frame right after wrapper header
    w.filename_or_fatbins = NULL;

    void **handle = p_reg_fatbin(&w);
    if (!handle) {
        fprintf(stderr, "__cudaRegisterFatBinary returned NULL\n");
        return 1;
    }
    printf("registered fatbin, handle=%p\n", (void *)handle);

    const char *kname = "_Z18identifyTileRangesiPmP5uint2";
    reg_func(handle, (const void *)identifyTileRanges_hoststub, (char *)kname,
             kname, -1, NULL, NULL, NULL, NULL, NULL);
    printf("registered kernel %s\n", kname);

    // Device buffers via the runtime API. cudaMemcpyKind: 1=HtoD, 2=DtoH.
    uint64_t keys[L];
    for (int i = 0; i < L; i++)
        keys[i] = ((uint64_t)tiles[i] << 32) | (uint64_t)i;

    void *d_keys = NULL;
    rc = p_malloc(&d_keys, L * sizeof(uint64_t));
    if (rc != 0) die("cudaMalloc keys", rc);
    void *d_ranges = NULL;
    rc = p_malloc(&d_ranges, NRANGE * sizeof(uint2c));
    if (rc != 0) die("cudaMalloc ranges", rc);

    uint32_t zeros[NRANGE * 2];
    memset(zeros, 0, sizeof(zeros));
    rc = p_memcpy(d_keys, keys, L * sizeof(uint64_t), 1);
    if (rc != 0) die("cudaMemcpy HtoD keys", rc);
    rc = p_memcpy(d_ranges, zeros, sizeof(zeros), 1);
    if (rc != 0) die("cudaMemcpy HtoD ranges", rc);

    // THE test: launch through cudaLaunchKernel with only the host stub
    // pointer. libcudart must resolve it to the registered module itself.
    int L_arg = L;
    void *args[3] = {&L_arg, &d_keys, &d_ranges};
    dim3s grid = {1, 1, 1};
    dim3s block = {256, 1, 1};
    rc = p_launch((const void *)identifyTileRanges_hoststub, grid, block,
                  args, 0, NULL);
    if (rc != 0) {
        fprintf(stderr, "cudaLaunchKernel failed: %d (%s)\n", rc,
                p_errstr(rc));
        return 1;
    }
    printf("cudaLaunchKernel OK\n");

    uint2c ranges[NRANGE];
    rc = p_memcpy(ranges, d_ranges, sizeof(ranges), 2);
    if (rc != 0) die("cudaMemcpy DtoH ranges", rc);

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

    p_free(d_keys);
    p_free(d_ranges);
    printf(ok ? "RUNTIME LAUNCH TEST OK\n" : "RUNTIME LAUNCH TEST FAILED\n");
    return ok ? 0 : 1;
}

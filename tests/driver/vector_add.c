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

int main() {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction kernel;
    CUdeviceptr d_a, d_b, d_c;
    const int N = 1024;
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) { a[i] = i * 1.0f; b[i] = i * 2.0f; }

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));

    char name[128];
    CHECK(cuDeviceGetName(name, sizeof(name), dev));
    printf("device: %s\n", name);



    CHECK(cuCtxCreate(&ctx, 0, dev));

    FILE* f = fopen("vectorAdd_11.ptx", "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1);
    fread(ptx, 1, sz, f);
    ptx[sz] = 0;
    fclose(f);

    CHECK(cuModuleLoadData(&mod, ptx));
    CHECK(cuModuleGetFunction(&kernel, mod, "_Z9vectorAddPKfS0_Pfi"));

    CHECK(cuMemAlloc(&d_a, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_b, N * sizeof(float)));
    CHECK(cuMemAlloc(&d_c, N * sizeof(float)));

    CHECK(cuMemcpyHtoD(d_a, a, N * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_b, b, N * sizeof(float)));

    void* args[4] = { &d_a, &d_b, &d_c, &(int){N} };
    CHECK(cuLaunchKernel(kernel, N/256, 1, 1, 256, 1, 1, 0, 0, args, NULL));

    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(c, d_c, N * sizeof(float)));

    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) { ok = 0; if (i < 10) printf("mismatch[%d]: %f != %f\n", i, c[i], a[i]+b[i]); }
    }
    printf(ok ? "VECTOR ADD OK: %d elements\n" : "VECTOR ADD FAILED\n", N);

    CHECK(cuMemFree(d_a));
    CHECK(cuMemFree(d_b));
    CHECK(cuMemFree(d_c));
    CHECK(cuModuleUnload(mod));
    CHECK(cuCtxDestroy(ctx));
    free(a); free(b); free(c);
    return ok ? 0 : 1;
}

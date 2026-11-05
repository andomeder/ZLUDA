/* A kernel whose shared window is a sized `.extern .shared` symbol: the declared
   1024 bytes are the static part, and each of the 1024 threads writes four more
   at 1024 + 4*tid. That is the shape a lifted kernel has, and the launch only
   asks for the dynamic part (4096 bytes), so the driver has to request the
   declared prefix on top of it. Without that, everything past 4096 bytes lands
   outside the window and the top of the array comes back as garbage.

   Expected output: out[tid] == tid + 1 for all 1024 threads. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS 1024
#define STATIC_BYTES 1024

int main(void) {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    FILE *file = fopen("shared_prefix_test.ptx", "rb");
    if (!file) {
        printf("cannot open shared_prefix_test.ptx\n");
        return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = malloc(size + 1);
    if (fread(text, 1, size, file) != (size_t)size) {
        return 1;
    }
    text[size] = 0;
    fclose(file);

    CUmodule module;
    CUresult result = cuModuleLoadData(&module, text);
    if (result) {
        printf("module load: %d\n", (int)result);
        return 1;
    }
    CUfunction kernel;
    if (cuModuleGetFunction(&kernel, module, "shared_prefix")) {
        printf("module build rejected: %d\n", (int)result);
        return 1;
    }

    CUdeviceptr out;
    unsigned int poison = 0xffffffffu;
    unsigned int host[THREADS];
    cuMemAlloc(&out, THREADS * sizeof(unsigned int));
    for (int i = 0; i < THREADS; i++) {
        cuMemcpyHtoD(out + i * sizeof(unsigned int), &poison, sizeof(poison));
    }

    void *arguments[1] = {&out};
    // Only the dynamic part: the static prefix is the driver's business.
    unsigned int dynamic_shared = THREADS * sizeof(unsigned int);
    result = cuLaunchKernel(kernel, 1, 1, 1, THREADS, 1, 1, dynamic_shared, 0, arguments, NULL);
    if (result) {
        printf("launch: %d\n", (int)result);
        return 1;
    }
    result = cuCtxSynchronize();
    if (result) {
        printf("sync: %d\n", (int)result);
        return 1;
    }
    cuMemcpyDtoH(host, out, THREADS * sizeof(unsigned int));

    int wrong = 0;
    for (int i = 0; i < THREADS; i++) {
        if (host[i] != (unsigned int)(i + 1)) {
            if (wrong < 4) {
                printf("  [%d] got %u want %d\n", i, host[i], i + 1);
            }
            wrong++;
        }
    }
    printf("SHARED WINDOW TEST: %d/%d wrong (static %d + dynamic %u bytes) %s\n",
           wrong, THREADS, STATIC_BYTES, dynamic_shared, wrong ? "FAILED" : "OK");
    return wrong ? 1 : 0;
}

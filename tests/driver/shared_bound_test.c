/* The shared window a launch asks for, when the kernel's PTX proves how much of
   it is reachable.

   Both kernels here declare a 32 KB `.extern .shared` array, which is the size
   a lifted kernel inherits from its whole module rather than from itself. The
   first addresses the window only at constant offsets, so the driver can reduce
   the request to what the kernel reaches; the second indexes it by blockIdx,
   which no launch-independent bound covers, so the driver keeps the declared
   size. Both must compute the same value either way: a window that is too small
   is silent corruption, so the numbers are what this test is for.

   Print the windows with ZLUDA_TRACE_SHARED=1 to see which case each kernel is
   in.

   Expected: "const 63" and "ctaid 7", and with ZLUDA_SHARED_PREFIX_MAX=0 the
   first kernel's window loses its declared part and the test fails. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        printf("cannot open %s\n", path);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = malloc(size + 1);
    if (!text || fread(text, 1, size, file) != (size_t)size) {
        return NULL;
    }
    text[size] = 0;
    fclose(file);
    return text;
}

int main(void) {
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    char *text = read_file("shared_bound_test.ptx");
    if (!text) {
        return 1;
    }
    CUmodule module;
    CUresult result = cuModuleLoadData(&module, text);
    if (result) {
        printf("module load failed: %d\n", result);
        return 1;
    }

    int failures = 0;
    struct {
        const char *name;
        int expected;
        int grid;
    } cases[] = {
        // 0 + 1 + 2 + 3 written to the last two words: 2 + 3
        {"shared_bound_const", 5, 1},
        {"shared_bound_ctaid", 7, 4},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        CUfunction function;
        result = cuModuleGetFunction(&function, module, cases[i].name);
        if (result) {
            printf("%s: not found (%d)\n", cases[i].name, result);
            failures++;
            continue;
        }
        CUdeviceptr out;
        if (cuMemAlloc(&out, sizeof(int))) {
            return 1;
        }
        int zero = 0;
        cuMemcpyHtoD(out, &zero, sizeof(int));
        void *arguments[] = {&out};
        result = cuLaunchKernel(function, cases[i].grid, 1, 1, 32, 1, 1,
                                /*sharedMemBytes=*/0, NULL, arguments, NULL);
        if (result) {
            printf("%s: launch failed (%d)\n", cases[i].name, result);
            failures++;
            continue;
        }
        if (cuCtxSynchronize()) {
            printf("%s: sync failed\n", cases[i].name);
            failures++;
            continue;
        }
        int value = -1;
        cuMemcpyDtoH(&value, out, sizeof(int));
        cuMemFree(out);
        int ok = value == cases[i].expected;
        printf("%s: %d (want %d) %s\n", cases[i].name, value, cases[i].expected,
               ok ? "OK" : "WRONG");
        if (!ok) {
            failures++;
        }
    }
    cuModuleUnload(module);
    free(text);
    printf("SHARED BOUND TEST: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}

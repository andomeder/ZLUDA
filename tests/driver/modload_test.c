/* What does cuModuleLoadData answer for an image it cannot translate?

   torch builds twelve of its ops (exp2, erfc, erfinv, i0, sinc, digamma, lgamma,
   mvlgamma, angle, prod, gcd, lcm) as CUDA source compiled at run time, and the
   runtime hands the result to the driver as a bare cubin. A cubin is not PTX
   text, so reading it as a C string finds no `.entry`, and the load used to
   report CUDA_SUCCESS with a module that had no kernels at all - the real
   failure then surfaced at cuModuleGetFunction as CUDA_ERROR_NOT_FOUND, which
   reads like a missing symbol rather than an image that was never loaded.

   Four images, each with a definite answer:

     a 26-byte text file          refused; it defines no kernel
     valid PTX                    loads, and its kernel is found
     a bare ELF cubin             refused with CUDA_ERROR_NOT_SUPPORTED when the
                                  SASS lift is off, loaded when it is on
     an all-zero 64-byte buffer    refused

   The point of arms 1 and 3 is that a SUCCESS with zero kernels is a lie: the
   first asserts the refusal, the second asserts that a cubin is recognised as
   compiled code rather than parsed as text (which is why it can say NOT_SUPPORTED
   instead of INVALID_PTX). */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char garbage[] = "not ptx at all, just bytes";

static const char ptx[] =
    ".version 8.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry modload() { ret; }\n";

/* A little-endian 64-bit ELF header with a section table that ends at 0x1000, so
   the image describes its own extent. Nothing else has to be valid: the driver
   refuses a cubin before it looks inside one when the lift is off, which is what
   makes this arm deterministic. */
static void make_elf(unsigned char *image) {
    memset(image, 0, 0x40);
    image[0] = 0x7f;
    image[1] = 'E';
    image[2] = 'L';
    image[3] = 'F';
    image[4] = 2; /* EI_CLASS 64-bit */
    image[5] = 1; /* EI_DATA little endian */
    image[6] = 1; /* EI_VERSION */
    image[0x10] = 2; /* ET_EXEC */
    image[0x12] = 0xbe; /* EM_CUDA */
    image[0x13] = 0x00;
    /* e_shoff = 0x80, e_shentsize = 0x40, e_shnum = 0x20 -> ends at 0x880 */
    image[0x28] = 0x80;
    image[0x3a] = 0x40;
    image[0x3c] = 0x20;
    /* e_flags = sm_86 in both the low and the third byte */
    image[0x30] = 0x56;
    image[0x32] = 0x56;
}

int main(void) {
    int failures = 0;
    cuInit(0);
    CUdevice device;
    cuDeviceGet(&device, 0);
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    /* arm 1: text that is not PTX */
    CUmodule module = NULL;
    CUresult result = cuModuleLoadData(&module, garbage);
    printf("garbage (%zu bytes)          -> %d\n", sizeof(garbage) - 1, (int)result);
    if (result == CUDA_SUCCESS) {
        printf("  FAIL  an image with no kernel reported a successful load\n");
        failures += 1;
    } else if (module != NULL) {
        printf("  FAIL  a refused load still handed out a module\n");
        failures += 1;
    } else {
        printf("  ok    refused\n");
    }

    /* arm 2: valid PTX still loads, and the kernel is really there */
    module = NULL;
    result = cuModuleLoadData(&module, ptx);
    printf("valid PTX                    -> %d\n", (int)result);
    CUfunction function = NULL;
    CUresult lookup = CUDA_ERROR_UNKNOWN;
    if (result == CUDA_SUCCESS) {
        lookup = cuModuleGetFunction(&function, module, "modload");
        printf("  cuModuleGetFunction        -> %d\n", (int)lookup);
    }
    if (result != CUDA_SUCCESS || lookup != CUDA_SUCCESS) {
        printf("  FAIL  valid PTX did not load and resolve\n");
        failures += 1;
    } else {
        printf("  ok    loaded and resolved\n");
    }

    /* arm 3: a bare cubin. The lift is off for this arm so the answer does not
       depend on whether a lifter is installed; the contract being tested is that
       compiled code is recognised and refused, never silently accepted empty. */
    unsigned char elf[0x40];
    make_elf(elf);
    unsetenv("ZLUDA_SASS_LIFT");
    module = NULL;
    result = cuModuleLoadData(&module, elf);
    printf("bare ELF, lift off           -> %d\n", (int)result);
    if (result == CUDA_SUCCESS) {
        printf("  FAIL  a cubin reported a successful load with no kernels\n");
        failures += 1;
    } else if (result != CUDA_ERROR_NOT_SUPPORTED) {
        printf("  FAIL  a cubin was not recognised as compiled code (%d)\n", (int)result);
        failures += 1;
    } else {
        printf("  ok    refused as compiled code\n");
    }

    /* arm 4: an image with nothing in it at all */
    unsigned char zeros[0x40];
    memset(zeros, 0, sizeof(zeros));
    module = NULL;
    result = cuModuleLoadData(&module, zeros);
    printf("64 zero bytes                -> %d\n", (int)result);
    if (result == CUDA_SUCCESS) {
        printf("  FAIL  an empty image reported a successful load\n");
        failures += 1;
    } else {
        printf("  ok    refused\n");
    }

    printf("\nMODLOAD TEST: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}

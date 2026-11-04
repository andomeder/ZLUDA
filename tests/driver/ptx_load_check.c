#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ptx>\n", argv[0]); return 2; }
    CUdevice dev; CUcontext ctx; CUmodule mod;
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1); fread(ptx, 1, sz, f); ptx[sz] = 0; fclose(f);
    CUresult r = cuModuleLoadData(&mod, ptx);
    if (r == CUDA_SUCCESS) { printf("LOAD OK: %s\n", argv[1]); return 0; }
    const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s);
    printf("LOAD FAIL: %s -> %s %s\n", argv[1], n, s);
    return 1;
}

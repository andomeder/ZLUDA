#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main() {
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[32]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("tid_test.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"tid_test"));
    CHECK(cuMemAlloc(&dout,128));
    CHECK(cuMemcpyHtoD(dout,out,128));
    void* args[1]={&dout};
    /* 1 block, 4 threads; thread t writes out[t] */
    CHECK(cuLaunchKernel(k,1,1,1,4,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,128));
    int wrong = 0;
    for (int i=0;i<4;i++) {
        if (out[i] != (unsigned)i) wrong++;
    }
    printf("TID TEST: %d/4 threads wrong, values %u %u %u %u\n",
           wrong, out[0], out[1], out[2], out[3]);
    return wrong ? 1 : 0;
}

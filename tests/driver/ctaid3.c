#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main() {
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[32]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaid3.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaid3"));
    CHECK(cuMemAlloc(&dout,128));
    CHECK(cuMemcpyHtoD(dout,out,128));
    void* args[1]={&dout};
    CHECK(cuLaunchKernel(k,8,1,1,1,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,128));
    int wrong = 0;
    for (int i=0;i<8;i++) {
        if (out[i] != (unsigned)i) wrong++;
    }
    printf("CTAID.X TEST: %d/8 blocks wrong, values %u %u %u %u %u %u %u %u\n",
           wrong, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    return wrong ? 1 : 0;
}

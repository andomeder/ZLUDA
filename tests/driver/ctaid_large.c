#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    int gx = argc>1?atoi(argv[1]):16;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[512]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaid3.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaid3"));
    CHECK(cuMemAlloc(&dout,2048));
    CHECK(cuMemcpyHtoD(dout,out,2048));
    void* args[1]={&dout};
    CHECK(cuLaunchKernel(k,gx,1,1,4,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,2048));
    printf("grid %dx1, %d blocks: ", gx, gx);
    for (int i=0;i<(gx<16?gx:16);i++) printf("%u ", out[i]);
    printf("\n");
    return 0;
}

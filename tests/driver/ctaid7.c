#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    int gx = argc>1?atoi(argv[1]):4;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[64]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaid7.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaid7"));
    CHECK(cuMemAlloc(&dout,256));
    CHECK(cuMemcpyHtoD(dout,out,256));
    void* args[1]={&dout};
    /* each block writes raw ctaid.x to out[0] and ctaid.x*32 to out[1..2] (u64) */
    CHECK(cuLaunchKernel(k,gx,1,1,1,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,256));
    printf("grid %dx1 (last block wins at out[0]): raw=%u  mul(ctaid.x,32)=%u\n",
        gx, out[0], out[1]);
    return 0;
}

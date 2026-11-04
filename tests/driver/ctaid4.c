#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    int gx = argc>1?atoi(argv[1]):2, gy = argc>2?atoi(argv[2]):1;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[256]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaid_dump.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaid_dump"));
    CHECK(cuMemAlloc(&dout,1024));
    CHECK(cuMemcpyHtoD(dout,out,1024));
    void* args[1]={&dout};
    /* kernel writes ctaid.x to byte offset ctaid.x*32 (= u32 index ctaid.x*8) */
    CHECK(cuLaunchKernel(k,gx,gy,1,4,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,1024));
    printf("grid %dx%d: ", gx, gy);
    for (int i=0;i<gx*gy;i++) printf("b%d=%u ", i, out[i*8]);
    printf("\n");
    return 0;
}

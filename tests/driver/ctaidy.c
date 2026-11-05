#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    int gx = argc>1?atoi(argv[1]):1, gy = argc>2?atoi(argv[2]):4;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[64]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaidy.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaidy"));
    CHECK(cuMemAlloc(&dout,256));
    CHECK(cuMemcpyHtoD(dout,out,256));
    void* args[1]={&dout};
    CHECK(cuLaunchKernel(k,gx,gy,1,1,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,256));
    int wrong = 0;
    for (int i=0;i<gy;i++) {
        if (out[i] != (unsigned)i) wrong++;
    }
    printf("CTAID.Y TEST: grid %dx%d, %d/%d rows wrong, values %u %u %u %u\n",
           gx, gy, wrong, gy, out[0], out[1], out[2], out[3]);
    return wrong ? 1 : 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    int gx = argc>1?atoi(argv[1]):3, gy = argc>2?atoi(argv[2]):2;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[8]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaidrace.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaidrace"));
    CHECK(cuMemAlloc(&dout,32));
    CHECK(cuMemcpyHtoD(dout,out,32));
    void* args[1]={&dout};
    CHECK(cuLaunchKernel(k,gx,gy,1,1,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,32));
    /* all blocks race; last block wins. Expected last block = (gx-1, gy-1) */
    printf("grid %dx%d, all blocks write slot 0: (ctaid.x, ctaid.y) = (%u, %u)\n",
        gx, gy, out[0], out[1]);
    printf("  expect last block (x=%d, y=%d)\n", gx-1, gy-1);
    return 0;
}

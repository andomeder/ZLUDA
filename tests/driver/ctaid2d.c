#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main(int argc, char** argv) {
    int gx = argc>1?atoi(argv[1]):3, gy = argc>2?atoi(argv[2]):2;
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr dout;
    unsigned int out[256]={0};
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("ctaid2d.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"ctaid2d"));
    CHECK(cuMemAlloc(&dout,1024));
    CHECK(cuMemcpyHtoD(dout,out,1024));
    void* args[1]={&dout};
    CHECK(cuLaunchKernel(k,gx,gy,1,1,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(out,dout,1024));
    /* kernel: block (x,y) writes x at u32 idx y*16+x*2, y at +1 */
    printf("grid %dx%d:\n", gx, gy);
    int ok = 1;
    for (int y=0;y<gy;y++) {
        for (int x=0;x<gx;x++) {
            int xi = y*16 + x*2;
            int rx = out[xi], ry = out[xi+1];
            if (rx != x || ry != y) ok = 0;
            printf("  (%d,%d)=(%u,%u)%s\n", x, y, rx, ry, (rx==x&&ry==y)?"":"  <-- WRONG");
        }
    }
    printf(ok ? "2D CTAID OK\n" : "2D CTAID FAIL\n");
    return ok?0:1;
}

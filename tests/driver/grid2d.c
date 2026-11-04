#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)
int main() {
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction k;
    CUdeviceptr da, db, dc;
    const int N = 65536;
    float *a=malloc(N*4), *b=malloc(N*4), *c=malloc(N*4);
    for (int i=0;i<N;i++){ a[i]=i*1.0f; b[i]=1.0f; }
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));
    FILE* f=fopen("vectorAdd2d_11.ptx","rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char* ptx=malloc(sz+1); fread(ptx,1,sz,f); ptx[sz]=0; fclose(f);
    CHECK(cuModuleLoadData(&mod,ptx));
    CHECK(cuModuleGetFunction(&k,mod,"_Z9vectorAddPKfS0_Pfi"));
    CHECK(cuMemAlloc(&da,N*4)); CHECK(cuMemAlloc(&db,N*4)); CHECK(cuMemAlloc(&dc,N*4));
    CHECK(cuMemcpyHtoD(da,a,N*4)); CHECK(cuMemcpyHtoD(db,b,N*4));
    void* args[4]={&da,&db,&dc,&(int){N}};
    /* 2D grid: 16x16 blocks, 256 threads/block */
    CHECK(cuLaunchKernel(k,16,16,1,256,1,1,0,0,args,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH(c,dc,N*4));
    int bad=0; for(int i=0;i<N;i++) if(c[i]!=a[i]+b[i]){bad++; if(bad<5) printf("mismatch[%d] %f != %f\n",i,c[i],a[i]+b[i]);}
    printf(bad? "GRID2D FAIL: %d bad\n":"GRID2D OK: 2D grid (%d threads) correct\n",bad?bad:N,N);
    return bad?1:0;
}

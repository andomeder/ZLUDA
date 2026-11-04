#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)

static CUmodule load_mod(CUcontext ctx, const char* ptxname) {
    CUmodule mod;
    FILE* f = fopen(ptxname, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ptxname); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* ptx = malloc(sz + 1); fread(ptx, 1, sz, f); ptx[sz] = 0; fclose(f);
    CHECK(cuModuleLoadData(&mod, ptx));
    free(ptx);
    return mod;
}

static void run_vector_add(CUfunction k, CUdeviceptr da, CUdeviceptr db, CUdeviceptr dc, int n) {
    int N = n;
    void* args[4] = { &da, &db, &dc, &N };
    CHECK(cuLaunchKernel(k, (n+255)/256, 1,1, 256,1,1, 0, 0, args, NULL));
}

int main(void) {
    CUdevice dev; CUcontext ctx;
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));

    /* ---- Test 1: cuMemGetInfo_v2 ---- */
    {
        size_t free_mem = 0, total_mem = 0;
        CHECK(cuMemGetInfo_v2(&free_mem, &total_mem));
        printf("MEM INFO: free=%zu total=%zu %s\n", free_mem, total_mem,
               (total_mem > 0 && free_mem <= total_mem)?"OK":"FAIL");
        if (total_mem == 0 || free_mem > total_mem) return 1;
    }

    /* ---- Test 2: host-pinned memory (cuMemAllocHost + device access) ---- */
    {
        CUmodule mod = load_mod(ctx, "vectorAdd_11.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "_Z9vectorAddPKfS0_Pfi"));
        const int N = 4096;
        float *ha = NULL, *hb = NULL, *hc = NULL;
        CHECK(cuMemAllocHost((void**)&ha, N * sizeof(float)));
        CHECK(cuMemAllocHost((void**)&hb, N * sizeof(float)));
        CHECK(cuMemAllocHost((void**)&hc, N * sizeof(float)));
        for (int i = 0; i < N; i++) { ha[i] = i * 1.0f; hb[i] = i * 2.0f; }
        /* device pointers for pinned host memory */
        CUdeviceptr da, db, dc;
        CHECK(cuMemHostGetDevicePointer_v2(&da, ha, 0));
        CHECK(cuMemHostGetDevicePointer_v2(&db, hb, 0));
        CHECK(cuMemHostGetDevicePointer_v2(&dc, hc, 0));
        /* direct HtoD of host pinned memory - no cuMemAlloc needed */
        run_vector_add(k, da, db, dc, N);
        CHECK(cuCtxSynchronize());
        int ok = 1;
        for (int i = 0; i < N; i++) {
            float expect = i * 3.0f;
            if (hc[i] != expect) { ok = 0; printf("  hc[%d]=%f expect %f\n", i, hc[i], expect); break; }
        }
        printf("PINNED HOST MEM: vector add on host-pinned ptrs %s\n", ok?"OK":"FAIL");
        CHECK(cuMemFreeHost(ha)); CHECK(cuMemFreeHost(hb)); CHECK(cuMemFreeHost(hc));
        if (!ok) return 1;
    }

    /* ---- Test 3: events (create/record/query/synchronize/elapsed) ---- */
    {
        CUevent start, end;
        CHECK(cuEventCreate(&start, 0));
        CHECK(cuEventCreate(&end, 0));
        /* record on default stream */
        CHECK(cuEventRecord(start, 0));
        CUresult r = cuEventQuery(start);
        if (r != CUDA_SUCCESS && r != CUDA_ERROR_NOT_READY) {
            printf("EVENT QUERY: unexpected error %d\n", r);
            return 1;
        }
        CHECK(cuEventSynchronize(start));
        /* run some work between events */
        CHECK(cuEventRecord(start, 0));
        CUmodule mod = load_mod(ctx, "vectorAdd_11.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "_Z9vectorAddPKfS0_Pfi"));
        CUdeviceptr da, db, dc;
        CHECK(cuMemAlloc(&da, 1024*sizeof(float)));
        CHECK(cuMemAlloc(&db, 1024*sizeof(float)));
        CHECK(cuMemAlloc(&dc, 1024*sizeof(float)));
        CHECK(cuMemsetD8_v2(da, 1, 1024*sizeof(float)));
        CHECK(cuMemsetD8_v2(db, 2, 1024*sizeof(float)));
        run_vector_add(k, da, db, dc, 1024);
        CHECK(cuEventRecord(end, 0));
        float ms = 0;
        CHECK(cuEventElapsedTime(&ms, start, end));
        CHECK(cuEventSynchronize(end));
        CHECK(cuEventElapsedTime(&ms, start, end));
        printf("EVENT ELAPSED: %f ms %s\n", ms, (ms >= 0.0f)?"OK":"FAIL");
        /* re-query after sync should be SUCCESS */
        CHECK(cuEventQuery(end));
        printf("EVENT QUERY after sync: OK\n");
        CHECK(cuEventDestroy_v2(start));
        CHECK(cuEventDestroy_v2(end));
        if (ms < 0.0f) return 1;
    }

    /* ---- Test 4: cuStreamQuery ---- */
    {
        CUstream stream;
        CHECK(cuStreamCreate(&stream, 0));
        CUmodule mod = load_mod(ctx, "vectorAdd_11.ptx");
        CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "_Z9vectorAddPKfS0_Pfi"));
        CUdeviceptr da, db, dc;
        CHECK(cuMemAlloc(&da, 256*sizeof(float)));
        CHECK(cuMemAlloc(&db, 256*sizeof(float)));
        CHECK(cuMemAlloc(&dc, 256*sizeof(float)));
        CHECK(cuMemsetD8_v2(da, 1, 256*sizeof(float)));
        CHECK(cuMemsetD8_v2(db, 2, 256*sizeof(float)));
        int N = 256;
        void* args[4] = { &da, &db, &dc, &N };
        CHECK(cuLaunchKernel(k, 1,1,1, 256,1,1, 0, stream, args, NULL));
        CUresult q = cuStreamQuery(stream);
        if (q != CUDA_SUCCESS && q != CUDA_ERROR_NOT_READY) {
            printf("STREAM QUERY: unexpected error %d\n", q);
            return 1;
        }
        CHECK(cuStreamSynchronize(stream));
        CHECK(cuStreamQuery(stream)); /* should be SUCCESS now */
        printf("STREAM QUERY: OK (busy->ready)\n");
        CHECK(cuStreamDestroy_v2(stream));
    }

    CHECK(cuCtxDestroy(ctx));
    printf("ALL EVENT/PINNED-MEM TESTS PASSED\n");
    return 0;
}

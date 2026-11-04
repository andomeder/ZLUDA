#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#define CHECK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* n; const char* s; cuGetErrorName(r,&n); cuGetErrorString(r,&s); fprintf(stderr,"FAIL %s:%d %s %s\n",__FILE__,__LINE__,n,s); exit(1);} } while(0)

/* ---- fatbin structs, mirroring zluda/src/impl/fatbin.rs ---- */
#pragma pack(push, 1)
struct fatbin_file_hdr {
    uint16_t kind;
    uint16_t version;
    uint32_t header_size;
    uint32_t payload_size;
    uint32_t unknown0;
    uint32_t compressed_size;
    uint32_t unknown1;
    uint32_t unknown2;
    uint32_t sm_version;
    uint32_t bit_width;
    uint32_t unknown3;
    uint64_t flags;
    uint64_t unknown5;
    uint64_t uncompressed_payload;
};
#pragma pack(pop)
struct fatbin_hdr { uint32_t magic; uint16_t version; uint16_t header_size; uint64_t files_size; };
struct fatbinc_wrapper { uint32_t magic; uint32_t version; void* data; void* filename_or_fatbins; };

#define FATBIN_MAGIC  0xBA55ED50u
#define FATBINC_MAGIC 0x466243B1u
#define FILE_KIND_PTX 1
#define FILE_VERSION  0x101
#define FLAG_64BIT    0x1

/* Build a plain fatbin (magic 0xBA55ED50) containing one PTX file record.
   Returns the total size. Caller provides buffers. */
static size_t build_fatbin(const char* ptx, size_t ptx_len,
                           struct fatbin_hdr** out_hdr, struct fatbin_file_hdr** out_file, char** out_ptx) {
    size_t file_hdr_size = sizeof(struct fatbin_file_hdr);
    size_t hdr_size = sizeof(struct fatbin_hdr);
    size_t files_size = file_hdr_size + ptx_len;
    struct fatbin_hdr* hdr = malloc(hdr_size + files_size);
    memset(hdr, 0, hdr_size + files_size);
    hdr->magic = FATBIN_MAGIC;
    hdr->version = 1;
    hdr->header_size = (uint16_t)hdr_size;
    hdr->files_size = files_size;
    struct fatbin_file_hdr* file = (struct fatbin_file_hdr*)((char*)hdr + hdr_size);
    file->kind = FILE_KIND_PTX;
    file->version = FILE_VERSION;
    file->header_size = (uint32_t)file_hdr_size;
    file->payload_size = (uint32_t)ptx_len;
    file->sm_version = 0x80;
    file->bit_width = 64;
    file->flags = FLAG_64BIT;
    file->uncompressed_payload = ptx_len;
    char* dst = (char*)file + file_hdr_size;
    memcpy(dst, ptx, ptx_len);
    *out_hdr = hdr;
    *out_file = file;
    *out_ptx = dst;
    return hdr_size + files_size;
}

/* Build a fatbinc wrapper (magic 0x466243B1, CUDA >= 11.2) around the same
   fatbin frame. Returns pointer to the wrapper (heap), sets *out_total. */
static struct fatbinc_wrapper* build_fatbinc(struct fatbin_hdr* frame, size_t frame_size, size_t* out_total) {
    struct fatbinc_wrapper* w = malloc(sizeof(*w) + frame_size);
    w->magic = FATBINC_MAGIC;
    w->version = 1;
    w->data = (void*)((char*)w + sizeof(*w));
    w->filename_or_fatbins = NULL;
    memcpy(w->data, frame, frame_size);
    *out_total = sizeof(*w) + frame_size;
    return w;
}

static size_t load_ptx(const char* path, char** out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = malloc(sz + 1); fread(buf, 1, sz, f); fclose(f);
    buf[sz] = 0;
    *out = buf;
    return (size_t)sz;
}

static unsigned run_atom_kernel(CUmodule mod) {
    CUfunction k; CHECK(cuModuleGetFunction(&k, mod, "atom_test"));
    CUdeviceptr dout; CHECK(cuMemAlloc(&dout, sizeof(unsigned)));
    unsigned zero = 0; CHECK(cuMemcpyHtoD(dout, &zero, sizeof(unsigned)));
    void* args[1] = { &dout };
    CHECK(cuLaunchKernel(k, 1,1,1, 256,1,1, 0, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    unsigned out = 0; CHECK(cuMemcpyDtoH(&out, dout, sizeof(unsigned)));
    CHECK(cuMemFree(dout));
    return out;
}

int main(void) {
    CUdevice dev; CUcontext ctx;
    CHECK(cuInit(0)); CHECK(cuDeviceGet(&dev,0)); CHECK(cuCtxCreate(&ctx,0,dev));

    char* ptx; size_t ptx_len = load_ptx("atom_test.ptx", &ptx);

    /* Test 1: plain fatbin (0xBA55ED50) via cuModuleLoadFatBinary */
    {
        struct fatbin_hdr* hdr; struct fatbin_file_hdr* file; char* p;
        size_t n = build_fatbin(ptx, ptx_len, &hdr, &file, &p);
        CUmodule mod;
        CHECK(cuModuleLoadFatBinary(&mod, hdr));
        unsigned out = run_atom_kernel(mod);
        printf("FATBIN (0xBA55ED50) via cuModuleLoadFatBinary: out=%u %s\n", out, out==256?"OK":"FAIL");
        if (out != 256) return 1;
        free(hdr);
    }

    /* Test 2: fatbinc wrapper (0x466243B1, CUDA >= 11.2 format) */
    {
        struct fatbin_hdr* hdr; struct fatbin_file_hdr* file; char* p;
        size_t frame_size = build_fatbin(ptx, ptx_len, &hdr, &file, &p);
        size_t total;
        struct fatbinc_wrapper* w = build_fatbinc(hdr, frame_size, &total);
        CUmodule mod;
        CHECK(cuModuleLoadFatBinary(&mod, w));
        unsigned out = run_atom_kernel(mod);
        printf("FATBINC (0x466243B1) via cuModuleLoadFatBinary: out=%u %s\n", out, out==256?"OK":"FAIL");
        if (out != 256) return 1;
        free(hdr); free(w);
    }

    CHECK(cuCtxDestroy(ctx));
    printf("ALL FATBIN TESTS PASSED\n");
    return 0;
}

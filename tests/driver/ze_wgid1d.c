// Pure Level Zero 1D WorkgroupId test (no ZLUDA).
// Kernel: ctaid3 = mov %r1, %ctaid.x; st [out], %r1 (all blocks race slot 0).
// Build: gcc -o ze_wgid1d ze_wgid1d.c -lze_loader
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <level_zero/ze_api.h>

#define CHECK(x) do { ze_result_t r = (x); if (r != ZE_RESULT_SUCCESS) { fprintf(stderr, "FAIL %s:%d ze=%d\n", __FILE__, __LINE__, r); exit(1);} } while(0)

int main(int argc, char** argv) {
    int gx = argc > 1 ? atoi(argv[1]) : 6;
    zeInit(ZE_INIT_FLAG_GPU_ONLY);
    uint32_t ndrivers = 0;
    CHECK(zeDriverGet(&ndrivers, NULL));
    ze_driver_handle_t driver;
    CHECK(zeDriverGet(&ndrivers, &driver));
    uint32_t ndevices = 0;
    CHECK(zeDeviceGet(driver, &ndevices, NULL));
    ze_device_handle_t device;
    CHECK(zeDeviceGet(driver, &ndevices, &device));
    ze_context_desc_t ctx_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
    ze_context_handle_t ctx;
    CHECK(zeContextCreate(driver, &ctx_desc, &ctx));

    FILE* f = fopen("/tmp/wgid1d.spv", "rb");
    if (!f) { perror("fopen spv"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* spv = malloc(sz);
    size_t n = fread(spv, 1, sz, f);
    fclose(f);

    ze_module_desc_t mod_desc;
    memset(&mod_desc, 0, sizeof(mod_desc));
    mod_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    mod_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    mod_desc.inputSize = (uint32_t)n;
    mod_desc.pInputModule = (const uint8_t*)spv;
    ze_module_handle_t mod;
    ze_module_build_log_handle_t log = NULL;
    ze_result_t mr = zeModuleCreate(ctx, device, &mod_desc, &mod, &log);
    if (mr != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "zeModuleCreate failed: %d\n", mr);
        if (log) { size_t lsz = 0; zeModuleBuildLogGetString(log, &lsz, NULL); char* buf = malloc(lsz); zeModuleBuildLogGetString(log, &lsz, buf); fprintf(stderr, "build log: %s\n", buf); }
        return 1;
    }

    ze_kernel_desc_t kdesc;
    memset(&kdesc, 0, sizeof(kdesc));
    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = "ctaid3";
    ze_kernel_handle_t k;
    CHECK(zeKernelCreate(mod, &kdesc, &k));
    CHECK(zeKernelSetGroupSize(k, 1, 1, 1));

    ze_host_mem_alloc_desc_t hdesc;
    memset(&hdesc, 0, sizeof(hdesc));
    hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void* dout;
    CHECK(zeMemAllocHost(ctx, &hdesc, 256, 64, &dout));
    memset(dout, 0, 256);
    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void*), &dout));

    ze_command_list_desc_t cldesc;
    memset(&cldesc, 0, sizeof(cldesc));
    cldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ze_command_list_handle_t cl;
    CHECK(zeCommandListCreate(ctx, device, &cldesc, &cl));
    ze_group_count_t gc = { (uint32_t)gx, 1, 1 };
    CHECK(zeCommandListAppendLaunchKernel(cl, k, &gc, NULL, 0, NULL));
    CHECK(zeCommandListClose(cl));
    ze_command_queue_desc_t qdesc;
    memset(&qdesc, 0, sizeof(qdesc));
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    ze_command_queue_handle_t q;
    CHECK(zeCommandQueueCreate(ctx, device, &qdesc, &q));
    CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL));
    CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));

    printf("pure L0 grid %dx1, last block wins slot 0: ctaid.x = %u (expect %d)\n", gx, *(unsigned int*)dout, gx - 1);
    return 0;
}

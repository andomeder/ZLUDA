// Pure Level Zero 2D WorkgroupId test (no ZLUDA).
// Uses the SPIR-V that ZLUDA generates for ctaid2d, launched directly via L0.
// Build: gcc -o ze_wgid2d ze_wgid2d.c -lze_loader
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <level_zero/ze_api.h>

#define CHECK(x) do { ze_result_t r = (x); if (r != ZE_RESULT_SUCCESS) { fprintf(stderr, "FAIL %s:%d ze=%d\n", __FILE__, __LINE__, r); exit(1);} } while(0)

int main() {
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

    FILE* f = fopen("/tmp/wgid2d.spv", "rb");
    if (!f) { perror("fopen spv"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* spv = malloc(sz);
    size_t n = fread(spv, 1, sz, f);
    fclose(f);
    printf("loaded %zu bytes SPIR-V\n", n);

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
        if (log) {
            size_t lsz = 0;
            zeModuleBuildLogGetString(log, &lsz, NULL);
            char* buf = malloc(lsz);
            zeModuleBuildLogGetString(log, &lsz, buf);
            fprintf(stderr, "build log: %s\n", buf);
        }
        return 1;
    }

    ze_kernel_desc_t kdesc;
    memset(&kdesc, 0, sizeof(kdesc));
    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = "ctaid2d";
    ze_kernel_handle_t k;
    CHECK(zeKernelCreate(mod, &kdesc, &k));
    CHECK(zeKernelSetGroupSize(k, 1, 1, 1));

    ze_host_mem_alloc_desc_t hdesc;
    memset(&hdesc, 0, sizeof(hdesc));
    hdesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void* dout;
    CHECK(zeMemAllocHost(ctx, &hdesc, 512, 64, &dout));
    CHECK(zeKernelSetArgumentValue(k, 0, sizeof(void*), &dout));

    ze_command_list_desc_t cldesc;
    memset(&cldesc, 0, sizeof(cldesc));
    cldesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    ze_command_list_handle_t cl;
    CHECK(zeCommandListCreate(ctx, device, &cldesc, &cl));
    memset(dout, 0, 512);
    ze_group_count_t gc = { 6, 1, 1 };
    CHECK(zeCommandListAppendLaunchKernel(cl, k, &gc, NULL, 0, NULL));
    CHECK(zeCommandListClose(cl));

    ze_command_queue_desc_t qdesc;
    memset(&qdesc, 0, sizeof(qdesc));
    qdesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    ze_command_queue_handle_t q;
    CHECK(zeCommandQueueCreate(ctx, device, &qdesc, &q));
    CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl, NULL));
    CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));

    unsigned int out[128] = {0};
    ze_command_list_handle_t cl2;
    CHECK(zeCommandListCreate(ctx, device, &cldesc, &cl2));
    memcpy(out, dout, 512);
    CHECK(zeCommandListClose(cl2));
    CHECK(zeCommandQueueExecuteCommandLists(q, 1, &cl2, NULL));
    CHECK(zeCommandQueueSynchronize(q, UINT64_MAX));

    printf("pure L0 grid 3x2 (kernel ctaid2d reads x,y writes per-block):\n");
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 3; x++) {
            int xi = y * 8 + x;
            printf("  (%d,%d)=(%u,%u)\n", x, y, out[xi], out[xi + 1]);
        }
    return 0;
}

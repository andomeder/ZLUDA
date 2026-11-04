#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <level_zero/ze_api.h>
#define CHECK(x) do { ze_result_t r = (x); if (r != ZE_RESULT_SUCCESS) { fprintf(stderr,"FAIL %s:%d err=%x\n",__FILE__,__LINE__,r); exit(1);} } while(0)
int main(void) {
    ze_driver_handle_t drv; ze_device_handle_t dev; ze_context_handle_t ctx;
    CHECK(zeInit(0));
    uint32_t nd = 0; CHECK(zeDriverGet(&nd, NULL));
    CHECK(zeDriverGet(&nd, &drv));
    uint32_t ndev = 0; CHECK(zeDeviceGet(drv, &ndev, NULL));
    CHECK(zeDeviceGet(drv, &ndev, &dev));
    ze_context_desc_t cdesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
    CHECK(zeContextCreate(drv, &cdesc, &ctx));

    /* event pool: host-visible */
    ze_event_pool_desc_t pdesc = { ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, NULL,
                                   ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1 };
    ze_event_pool_handle_t pool;
    CHECK(zeEventPoolCreate(ctx, &pdesc, 1, &dev, &pool));
    ze_event_desc_t edesc = { ZE_STRUCTURE_TYPE_EVENT_DESC, NULL, 0,
                              ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST };
    ze_event_handle_t ev;
    CHECK(zeEventCreate(pool, &edesc, &ev));

    /* immediate command list */
    ze_command_queue_desc_t qdesc = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL, 0, 0, 0,
                                      ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                                      ZE_COMMAND_QUEUE_PRIORITY_NORMAL };
    ze_command_list_handle_t cmdlist;
    CHECK(zeCommandListCreateImmediate(ctx, dev, &qdesc, &cmdlist));
    fprintf(stderr, "appending barrier...\n"); fflush(stderr);
    CHECK(zeCommandListAppendBarrier(cmdlist, ev, 0, NULL));
    fprintf(stderr, "barrier appended, host-syncing...\n"); fflush(stderr);
    CHECK(zeEventHostSynchronize(ev, UINT64_MAX));
    fprintf(stderr, "host sync done. querying...\n"); fflush(stderr);
    ze_result_t q = zeEventQueryStatus(ev);
    fprintf(stderr, "query=%x (0=success)\n", q);
    printf("ZE BARRIER TEST OK\n");
    return 0;
}

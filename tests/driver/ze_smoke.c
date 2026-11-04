#include <stdio.h>
#include <level_zero/ze_api.h>
int main() {
    zeInit(ZE_INIT_FLAG_GPU_ONLY);
    uint32_t nd = 0;
    zeDriverGet(&nd, NULL);
    printf("drivers: %u\n", nd);
    ze_driver_handle_t drv;
    zeDriverGet(&nd, &drv);
    uint32_t ndev = 0;
    zeDeviceGet(drv, &ndev, NULL);
    ze_device_handle_t dev;
    zeDeviceGet(drv, &ndev, &dev);
    ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    zeDeviceGetProperties(dev, &props);
    printf("device: %s\n", props.name);
    printf("deviceId: 0x%x\n", props.deviceId);
    printf("euCount: %u\n", props.numEUsPerSubslice);
    return 0;
}

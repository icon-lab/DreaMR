import time
import torch

from pynvml import *
nvmlInit()


def getGpuLoad(gpuId):

    h = nvmlDeviceGetHandleByIndex(gpuId)
    info = nvmlDeviceGetMemoryInfo(h)

    t = info.total
    f = info.free
    u = info.used

    return (u) / t


def getAvailableGpus(loadThreshold):

    gpuSamplingTime = 1  # seconds
    numberOfSamples = 5
    # loadThreshold = 0.5 # in order to call a gpu free, it should be at least 1-loadThreshold free

    numberOfGpus = torch.cuda.device_count()

    gpuAvailability = {}

    for i in range(numberOfGpus):
        gpuAvailability[i] = {"available": 1, "loadAverage": 0}

    for i in range(numberOfSamples):

        for gpuId in range(numberOfGpus):

            load = getGpuLoad(gpuId)

            gpuAvailability[gpuId]["available"] = gpuAvailability[gpuId]["available"] and (
                load < loadThreshold)
            gpuAvailability[gpuId]["loadAverage"] = (
                gpuAvailability[gpuId]["loadAverage"] * (i+1.0) + load)/(i+2.0)

        time.sleep(gpuSamplingTime)

    availableGpus = []
    for gpuId in gpuAvailability:
        if(gpuAvailability[gpuId]["available"]):
            availableGpus.append(
                (gpuId, gpuAvailability[gpuId]["loadAverage"]))

    print(availableGpus)

    availableGpus = [gpu[0]
                     for gpu in sorted(availableGpus, key=lambda x: x[1])]

    print(availableGpus)

    return availableGpus

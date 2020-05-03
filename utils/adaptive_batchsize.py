import os

import pynvml
from torch import cuda

from .tensor_section_functions import tensor_size, tensor_memory, tensor_cuda

GPUS = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]


def _get_free_memory_size():
    pynvml.nvmlInit()

    free_memory_size = 0
    for i in GPUS:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('free mem at gpu {0}: {1}'.format(i, meminfo.free))
        free_memory_size += meminfo.free - 1

    return max(free_memory_size, 0)


def get_max_batchsize(fun, *samples):
    samples = tensor_cuda(samples)

    gpu_num = cuda.device_count()
    sample_num = tensor_size(samples, dim=0)

    samples_memory = tensor_memory(samples)
    memory_per_sample = samples_memory // sample_num + 1

    for i in range(10):
        fun(*samples)  # warm up

    for i in range(gpu_num):
        cuda.reset_max_memory_cached(i)

    max_used_memory_pre = sum([cuda.max_memory_allocated(i) for i in range(gpu_num)])
    fun(*samples)
    max_used_memory_post = sum([cuda.max_memory_allocated(i) for i in range(gpu_num)])

    calling_memory_per_sample = (max_used_memory_post - max_used_memory_pre) // sample_num + 1

    total_memory = sum([cuda.memory_reserved(i) - 1 for i in range(gpu_num)]) + _get_free_memory_size() - 1
    used_memory = sum([cuda.memory_allocated(i) + 1 for i in range(gpu_num)]) - samples_memory + 1
    free_memory = total_memory - used_memory - 1

    max_batchsize = free_memory // (memory_per_sample + calling_memory_per_sample) - 1
    print('total_mem: {0}'.format(total_memory))
    print('used_mem: {0}'.format(used_memory))
    print('{0} // ({1} + {2}) - 1'.format(free_memory, memory_per_sample, calling_memory_per_sample))

    return max(max_batchsize, 1)

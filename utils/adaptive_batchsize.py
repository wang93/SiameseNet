import os

import pynvml
from torch import cuda

from .tensor_section_functions import tensor_size, tensor_memory, tensor_cuda, tensor_repeat, tensor_cpu

GPUS = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
GPU_NUM = cuda.device_count()


def get_free_memory_size():
    pynvml.nvmlInit()

    free_memory_size = 0
    for i in GPUS:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('free mem at gpu {0}: {1}'.format(i, meminfo.free))
        free_memory_size += meminfo.free - 1

    return max(free_memory_size, 0)


def get_memory_cost(fun, *samples):
    # warm up
    # for i in range(4):
    #     fun(*samples)

    for i in range(GPU_NUM):
        cuda.reset_max_memory_cached(i)

    max_used_memory_pre = sum([cuda.max_memory_allocated(i) for i in range(GPU_NUM)])
    fun(*samples)
    max_used_memory_post = sum([cuda.max_memory_allocated(i) for i in range(GPU_NUM)])

    memory_cost = max_used_memory_post - max_used_memory_pre

    return memory_cost


def get_max_batchsize(fun, *samples):
    samples = tensor_cpu(samples)

    total_memory = sum([cuda.memory_reserved(i) - 1 for i in range(GPU_NUM)]) + get_free_memory_size() - 1
    used_memory = sum([cuda.memory_allocated(i) + 1 for i in range(GPU_NUM)])
    free_memory = total_memory - used_memory - 1

    samples = tensor_cuda(samples)

    sample_memory = tensor_memory(samples)
    sample_num = tensor_size(samples, dim=0)
    memory_per_sample = sample_memory / sample_num

    memory_cost = get_memory_cost(fun, *samples)
    memory_cost_2x = get_memory_cost(fun, *tensor_repeat(samples, 0, 2))
    calling_memory_per_sample = (memory_cost_2x - memory_cost) // sample_num + 1
    calling_memory_base = max(memory_cost * 2 - memory_cost_2x, 1)

    max_batchsize = (free_memory - calling_memory_base) // (memory_per_sample + calling_memory_per_sample) - 1
    # print('total_mem: {0}'.format(total_memory))
    # print('used_mem: {0}'.format(used_memory))
    # print('calling memory base: {0}'.format(calling_memory_base))
    # print('{0} // ({1} + {2}) - 1'.format(free_memory, memory_per_sample, calling_memory_per_sample))
    print(max_batchsize)

    return int(max(max_batchsize, 1))

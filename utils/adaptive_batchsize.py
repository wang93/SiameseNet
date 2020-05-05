import math
import os

import pynvml
from torch import cuda

from .tensor_section_functions import tensor_size, tensor_memory, tensor_cuda, tensor_repeat

GPUS = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
GPU_NUM = len(GPUS)


def get_free_memory_size():
    pynvml.nvmlInit()

    free_memory_size = 0
    for i in GPUS:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('free mem at gpu {0}: {1}'.format(i, meminfo.free))
        free_memory_size += meminfo.free - 1

    return max(free_memory_size, 0)


def get_equal_free_memory_size():
    pynvml.nvmlInit()

    free_memory_size = []
    for i in GPUS:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('free mem at gpu {0}: {1}'.format(i, meminfo.free))
        free_memory_size.append(meminfo.free)

    equal_free_memory_size = min(free_memory_size) * len(free_memory_size)
    return max(equal_free_memory_size - 1, 0)


def get_memory_cost(fun, *samples):
    # warm up
    # fun(*samples)

    for i in range(GPU_NUM):
        cuda.reset_max_memory_allocated(i)

    max_used_memory_pre = sum([cuda.max_memory_allocated(i) for i in range(GPU_NUM)])
    fun(*samples)
    max_used_memory_post = sum([cuda.max_memory_allocated(i) for i in range(GPU_NUM)])

    memory_cost = max_used_memory_post - max_used_memory_pre

    return max(memory_cost, 1)


def get_max_batchsize(fun, *samples):
    samples = tensor_cuda(samples)

    sample_memory = tensor_memory(samples)
    sample_num = tensor_size(samples, dim=0)
    memory_per_sample = sample_memory / sample_num

    total_memory = sum([cuda.memory_reserved(i) - 1 for i in range(GPU_NUM)]) + get_free_memory_size() - 1
    used_memory = sum([cuda.memory_allocated(i) + 1 for i in range(GPU_NUM)])
    free_memory = total_memory - used_memory + sample_memory - 1

    memory_cost = get_memory_cost(fun, *samples)
    memory_cost_2x = get_memory_cost(fun, *tensor_repeat(samples, 0, 2))
    calling_memory_per_sample = max((memory_cost_2x - memory_cost) // sample_num + 1, 1)
    calling_memory_base = max(memory_cost * 2 - memory_cost_2x, 1)

    max_batchsize = (free_memory - calling_memory_base) // (memory_per_sample + calling_memory_per_sample) - 1

    return int(max(max_batchsize, 1))


def get_max_equal_batchsize(fun, *samples):
    samples = tensor_cuda(samples)

    sample_memory = tensor_memory(samples)
    sample_num = tensor_size(samples, dim=0)
    memory_per_sample = sample_memory / sample_num

    cuda.empty_cache()
    free_memory = get_equal_free_memory_size() - 1

    memory_cost = get_memory_cost(fun, *samples)
    memory_cost_2x = get_memory_cost(fun, *tensor_repeat(samples, 0, 2))
    calling_memory_per_sample = max((memory_cost_2x - memory_cost) // sample_num + 1, 1)
    calling_memory_base = max(memory_cost * 2 - memory_cost_2x + 1, 1)

    max_batchsize = (free_memory - calling_memory_base) // (memory_per_sample + calling_memory_per_sample) - 1
    print(
        'max_batchsize{0} = (free_memory{1} - calling_memory_base{2}) // (memory_per_sample{3} + calling_memory_per_sample{4}) - 1'
        .format(max_batchsize, free_memory, calling_memory_base, memory_per_sample, calling_memory_per_sample))

    max_batchsize = (max_batchsize // GPU_NUM) * GPU_NUM

    return int(max(max_batchsize, 1))


def get_optimized_batchsize(fun, *samples):
    '''I am not sure that the returned batchsize can make the computaion faster'''
    max_batchsize = get_max_equal_batchsize(fun, *samples)
    per_gpu_max_batchsize = max(max_batchsize // GPU_NUM, 1)
    per_gpu_opti_batchsize = 2 ** int(math.log2(per_gpu_max_batchsize))
    optimized_batchsize = GPU_NUM * per_gpu_opti_batchsize

    return int(max(optimized_batchsize, 1))

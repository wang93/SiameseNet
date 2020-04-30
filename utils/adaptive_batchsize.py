from torch import cuda

from .tensor_section_functions import tensor_size, tensor_memory, tensor_cuda


def get_max_batchsize(fun, *samples):
    samples = tensor_cuda(samples)

    gpu_num = cuda.device_count()
    sample_num = tensor_size(samples, dim=0)

    samples_memory = tensor_memory(samples)
    memory_per_sample = samples_memory // sample_num + 1

    fun(*samples)  # warm up

    for i in range(gpu_num):
        cuda.reset_max_memory_cached(i)

    max_used_memory_pre = sum([cuda.max_memory_allocated(i) for i in range(gpu_num)])
    fun(*samples)
    max_used_memory_post = sum([cuda.max_memory_allocated(i) for i in range(gpu_num)])

    calling_memory_per_sample = (max_used_memory_post - max_used_memory_pre) // sample_num + 1

    total_memory = sum([cuda.memory_reserved(i) for i in range(gpu_num)])
    used_memory = sum([cuda.memory_allocated(i) for i in range(gpu_num)])
    free_memory = total_memory - used_memory

    max_batchsize = free_memory // (memory_per_sample + calling_memory_per_sample) - 1

    return max(max_batchsize, 1)

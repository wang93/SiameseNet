import torch
from torch import Tensor


def slice_tensor(data, indices):
    if isinstance(data, Tensor):
        return data[indices]
    elif isinstance(data, (list, tuple)):
        return [slice_tensor(d, indices) for d in data]
    elif isinstance(data, dict):
        return {k: slice_tensor(v, indices) for k, v in data.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def cat_tensor_pair(a, b, dim):
    assert type(a) == type(b)
    if isinstance(a, Tensor):
        return torch.cat((a, b), dim=dim)
    elif isinstance(a, (list, tuple)):
        return [cat_tensor_pair(i, j, dim) for i, j in zip(a, b)]
    elif isinstance(a, dict):
        return {k: cat_tensor_pair(v, b[k], dim) for k, v in a.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(a)))


def tensor_cpu(data):
    if isinstance(data, Tensor):
        return data.cpu()
    elif isinstance(data, (list, tuple)):
        return [tensor_cpu(d) for d in data]
    elif isinstance(data, dict):
        return {k: tensor_cpu(v) for k, v in data.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def tensor_cuda(data):
    if isinstance(data, Tensor):
        return data.cuda()
    elif isinstance(data, (list, tuple)):
        return [tensor_cuda(d) for d in data]
    elif isinstance(data, dict):
        return {k: tensor_cuda(v) for k, v in data.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def tensor_repeat(data, dim, num, interleave=False):
    if not interleave:
        fun = torch.Tensor.repeat
    else:
        fun = torch.Tensor.repeat_interleave

    if isinstance(data, Tensor):
        dim_num = len(data.size())
        szs = [1, ] * dim_num
        szs[dim] = num
        return fun(data, szs)
    elif isinstance(data, (list, tuple)):
        return [tensor_repeat(d, dim, num, interleave) for d in data]
    elif isinstance(data, dict):
        return {k: tensor_repeat(v, dim, num, interleave) for k, v in data.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def _all_same(l: list):
    for i in l:
        if i != l[0]:
            return False
    return True


def tensor_size(data, dim):
    if isinstance(data, Tensor):
        return data.size(dim)
    elif isinstance(data, (list, tuple)):
        results = [tensor_size(d, dim) for d in data]
        if _all_same(results):
            return results[0]
        else:
            raise ValueError('sizes of tensors are not consistent in dim {0}'.format(dim))
    elif isinstance(data, dict):
        results = [tensor_size(v, dim) for _, v in data.items()]
        if _all_same(results):
            return results[0]
        else:
            raise ValueError('sizes of tensors are not consistent in dim {0}'.format(dim))
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))

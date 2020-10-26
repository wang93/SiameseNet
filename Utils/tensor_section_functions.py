import torch
from torch import Tensor

__all__ = ['slice_tensor', 'split_tensor', 'cat_tensor_pair', 'cat_tensors',
           'tensor_cpu', 'tensor_cuda', 'tensor_repeat', 'tensor_size', 'dimidiation_tensor',
           'tensor_memory', 'tensor_attr', 'combine_tensor_pair', 'tensor_float']


def slice_tensor(data, indices):
    if isinstance(data, Tensor):
        return data[indices]
    elif isinstance(data, (list, tuple)):
        return [slice_tensor(d, indices) for d in data]
    elif isinstance(data, dict):
        return {k: slice_tensor(v, indices) for k, v in data.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def split_tensor(data, dim, split_size):
    if isinstance(data, Tensor):
        return torch.split(data, split_size, dim)
    elif isinstance(data, (list, tuple)):
        return [*zip(*[split_tensor(d, dim, split_size) for d in data])]
    elif isinstance(data, dict):
        vss = [split_tensor(v, dim, split_size) for _, v in data.items()]
        result = []
        for vs in zip(*vss):
            one_dict = {k: v for k, v in zip(data.keys(), vs)}
            result.append(one_dict)
        return result
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


def dimidiation_tensor(data, dim):
    if isinstance(data, Tensor):
        return torch.chunk(data, 2, dim=dim)
    elif isinstance(data, (list, tuple)):
        raise NotImplementedError
    elif isinstance(data, dict):
        raise NotImplementedError
    else:
        raise TypeError('type {0} is not supported'.format(type(a)))


def combine_tensor_pair(a, b):
    assert type(a) == type(b)
    if isinstance(a, Tensor):
        return (a, b)
    elif isinstance(a, (list, tuple)):
        return [combine_tensor_pair(i, j) for i, j in zip(a, b)]
    elif isinstance(a, dict):
        return {k: combine_tensor_pair(v, b[k]) for k, v in a.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(a)))


def cat_tensors(data, dim):
    if not isinstance(data, (list, tuple, dict)):
        raise ValueError

    if isinstance(data, dict):
        raise NotImplementedError

    if isinstance(data[0], Tensor):
        return torch.cat(data, dim=dim)
    elif isinstance(data[0], (list, tuple)):
        return [cat_tensors(d, dim) for d in zip(*data)]
    elif isinstance(data[0], dict):
        raise NotImplementedError
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


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


def tensor_float(data):
    if isinstance(data, Tensor):
        return data.float()
    elif isinstance(data, (list, tuple)):
        return [tensor_float(d) for d in data]
    elif isinstance(data, dict):
        return {k: tensor_float(v) for k, v in data.items()}
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def tensor_repeat(data, dim, num, interleave=False):
    if isinstance(data, Tensor):
        if not interleave:
            dim_num = len(data.size())
            szs = [1, ] * dim_num
            szs[dim] = num
            return torch.Tensor.repeat(data, szs)
        else:
            return torch.Tensor.repeat_interleave(data, repeats=num, dim=dim)

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


def tensor_memory(data):
    if isinstance(data, Tensor):
        return data.element_size() * data.nelement()
    elif isinstance(data, (list, tuple)):
        return sum([tensor_memory(d) for d in data])
    elif isinstance(data, dict):
        return sum([tensor_memory(v) for _, v in data.items()])
    else:
        raise TypeError('type {0} is not supported'.format(type(data)))


def tensor_attr(data, attr):
    if isinstance(data, Tensor):
        return getattr(data, attr)
    elif isinstance(data, (list, tuple)):
        results = [tensor_attr(d, attr) for d in data]
        if _all_same(results):
            return results[0]
        else:
            raise ValueError('{0} of tensors are not consistent!'.format(attr))

    elif isinstance(data, dict):
        results = [tensor_attr(v, attr) for _, v in data.items()]
        if _all_same(results):
            return results[0]
        else:
            raise ValueError('{0} of tensors are not consistent!'.format(attr))

    else:
        raise TypeError('type {0} is not supported'.format(type(data)))

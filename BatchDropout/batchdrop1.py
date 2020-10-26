# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/26 14:33

"""
the outputs of batchdrop are 0. or 1. (before scaling)
"""
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN
import torch


def expand(stat, target_size):
    if len(target_size) == 4:
        stat = stat.unsqueeze(1).unsqueeze(2).expand(target_size[1:])
    elif len(target_size) == 2:
        pass
    else:
        raise NotImplementedError
    return stat


class batch_dropout(Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x = args[0]
        mines = args[1]
        maxes = args[2]
        eps = args[3]

        sz = x.size()
        data = x.detach()

        intensity = (data - expand(mines, sz)) / expand(maxes - mines + eps, sz)
        rand = torch.rand_like(data, requires_grad=False)
        y = (intensity > rand).float().detach()
        ctx.save_for_backward(y)

        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_y = grad_outputs[0]
        y = ctx.saved_tensors[0]
        grad_x = grad_y * y

        return grad_x, None, None, None


class BatchDrop(origin_BN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        if not track_running_stats:
            raise NotImplementedError

        super(BatchDrop, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('running_max', torch.ones(num_features))
        self.register_buffer('running_min', -torch.ones(num_features))

        self.eps = self.eps ** 0.5

    @staticmethod
    def _max(data, dims, keepdim):
        for dim in dims:
            data, _ = data.max(dim=dim, keepdim=keepdim)
        return data

    @staticmethod
    def _min(data, dims, keepdim):
        for dim in dims:
            data, _ = data.min(dim=dim, keepdim=keepdim)
        return data

    def forward(self, input: torch.Tensor):
        sz = input.size()
        data = input.detach()
        if self.training:
            if input.dim() == 4:
                reduced_dim = (3, 2, 0)
            elif input.dim() == 2:
                reduced_dim = (0, )
            else:
                raise NotImplementedError

            cur_max = self._max(data, dims=reduced_dim, keepdim=False)
            cur_min = self._min(data, dims=reduced_dim, keepdim=False)
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * cur_max
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * cur_min

        else:
            cur_max = self.running_max
            cur_min = self.running_min

        y = batch_dropout.apply(input, cur_min, cur_max, self.eps)

        if self.affine:
            z = y * expand(self.weight, sz)
        else:
            z = y

        return z


def convert_model(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = torch.nn.DataParallel(mod, device_ids=module.device_ids)
        return mod

    mod = module
    if isinstance(module, origin_BN):
        mod = BatchDrop(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod
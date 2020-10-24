# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/24 17:29

"""
minimax dropout
"""
import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN


class _BatchDrop(origin_BN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        if not track_running_stats:
            raise NotImplementedError

        super(_BatchDrop, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.register_buffer('running_max', torch.ones(num_features))
        self.register_buffer('running_min', -torch.ones(num_features))

        self.eps = self.eps ** 0.5

    def _check_input_dim(self, input):
        raise NotImplementedError

    @staticmethod
    def expand(stat, target_size):
        raise NotImplementedError

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
        self._check_input_dim(input)

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

        intensity = (input - self.expand(cur_min, sz)) / self.expand(cur_max - cur_min + self.eps, sz)
        # if self.affine:
        #     intensity = intensity + self.expand(self.bias, sz)
        rand = torch.rand(*sz, dtype=input.dtype, device=input.device)
        mask = (intensity > rand).float().detach()
        y = intensity * mask

        if self.affine:
            z = y * self.expand(self.weight, sz)
        else:
            z = y

        return z


class BatchDrop1d(_BatchDrop):
    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    @staticmethod
    def expand(stat, *args, **kwargs):
        return stat


class BatchDrop2d(_BatchDrop):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    @staticmethod
    def expand(stat, target_size):
        stat = stat.unsqueeze(1).unsqueeze(2).expand(target_size[1:])
        return stat


def convert_model(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = torch.nn.DataParallel(mod, device_ids=module.device_ids)
        return mod

    mod = module
    for pth_module, id_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                      torch.nn.modules.batchnorm.BatchNorm2d],
                                     [BatchDrop1d,
                                      BatchDrop2d]):
        if isinstance(module, pth_module):
            mod = id_module(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod
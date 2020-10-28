# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/28 16:47
"""decreasing-momentum BN"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN


class _BatchNorm(origin_BN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.running_var = torch.zeros(num_features)

    @staticmethod
    def expand(stat, target_size):
        if len(target_size) == 4:
            stat = stat.unsqueeze(1).unsqueeze(2).expand(target_size[1:])
        elif len(target_size) == 2:
            pass
        else:
            raise NotImplementedError

        return stat

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        self._check_input_dim(input)

        sz = input.size()
        if self.training:
            if input.dim() == 4:
                reduced_dim = (0, 2, 3)
            elif input.dim() == 2:
                reduced_dim = (0,)
            else:
                raise NotImplementedError

            data = input.detach()

            di_mean = torch.mean(data, dim=reduced_dim, keepdim=False)
            di_var = torch.var(data, dim=reduced_dim, keepdim=False, unbiased=False)

            if self.track_running_stats:
                self.num_batches_tracked += 1
                cur_momentum = self.momentum + (1. - self.momentum) ** self.num_batches_tracked
                self.running_mean = (1. - cur_momentum) * self.running_mean + cur_momentum * di_mean
                self.running_var = (1. - cur_momentum) * self.running_var + cur_momentum * di_var

            else:
                raise NotImplementedError

        y = (input - self.expand(self.running_mean, sz)) \
            / self.expand(torch.sqrt(self.eps + self.running_var), sz)

        if self.affine:
            z = y * self.expand(self.weight, sz) + self.expand(self.bias, sz)
        else:
            z = y

        return z


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


def convert_model(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = torch.nn.DataParallel(mod, device_ids=module.device_ids)
        return mod

    mod = module
    for pth_module, id_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                      torch.nn.modules.batchnorm.BatchNorm2d],
                                     [BatchNorm1d,
                                      BatchNorm2d]):
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

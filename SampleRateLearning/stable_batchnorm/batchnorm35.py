# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/21 16:33

"""Classical BN with no bias"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN


class _BatchNorm(origin_BN):
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
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        sz = input.size()
        if self.training:
            if input.dim() == 4:
                reduced_dim = (0, 2, 3)
            elif input.dim() == 2:
                reduced_dim = (0, )
            else:
                raise NotImplementedError

            data = input.detach()

            di_mean = torch.mean(data, dim=reduced_dim, keepdim=False)
            di_var = torch.var(data, dim=reduced_dim, keepdim=False, unbiased=False)

            if self.track_running_stats:
                self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * di_mean
                self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * di_var

            else:
                self.running_mean = di_mean
                self.running_var = di_var

            y = (input - self.expand(di_mean, sz)) \
                / self.expand(torch.sqrt(self.eps + di_var), sz)

        else:
            y = (input - self.expand(self.running_mean, sz)) \
                / self.expand(torch.sqrt(self.eps + self.running_var), sz)

        if self.affine:
            z = y * self.expand(self.weight, sz)
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

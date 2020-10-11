# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/11 17:14

"""
for bi structure,
batch-wise estimation,
moving-average,
biased estimation,
bias-corrected,
stds via total running_mean,
.../(eps + std)
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN
from SampleRateLearning.stable_batchnorm import global_variables as batch_labels


class _BatchNorm(origin_BN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        if not track_running_stats:
            raise NotImplementedError

        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.running_var = torch.zeros(num_features)
        self.eps = pow(self.eps, 0.5)

        self.register_buffer('running_mean_ori', torch.zeros(num_features))
        self.register_buffer('running_std_ori', torch.zeros(num_features))

    def _check_input_dim(self, input):
        raise NotImplementedError

    @staticmethod
    def expand(stat, target_size):
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        self._check_input_dim(input)

        sz = input.size()
        if self.training:
            data = input.detach()
            self.num_batches_tracked += 1

            if input.dim() == 4:
                means = data.mean(dim=(0, 2, 3), keepdim=False)
            elif input.dim() == 2:
                means = data.mean(dim=0, keepdim=False)
            else:
                raise NotImplementedError

            correction_factor = (1. - (1. - self.momentum) ** self.num_batches_tracked)
            self.running_mean_ori = (1 - self.momentum) * self.running_mean_ori + self.momentum * means
            self.running_mean = self.running_mean_ori / correction_factor

            data = data - self.expand(self.running_mean, sz)

            if input.dim() == 4:
                stds = data.square().mean(dim=(0, 2, 3), keepdim=False).sqrt()
            elif input.dim() == 2:
                stds = data.square().mean(dim=0, keepdim=False).sqrt()

            self.running_std_ori = (1 - self.momentum) * self.running_std_ori+ self.momentum * stds

            # Note: the running_var is running_std indeed, for convenience of external calling, it has not been renamed.
            self.running_var = (self.running_std_ori / correction_factor)

        # Note: the running_var is running_stpd indeed, for convenience of external calling, it has not been renamed.
        y = (input - self.expand(self.running_mean, sz)) \
            / self.expand((self.running_var + self.eps), sz)

        if self.affine:
            z = y * self.expand(self.weight, sz) + self.expand(self.bias, sz)
        else:
            z = y

        return z


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))
        # if input.dim() != 2 and input.dim() != 3:
        #     raise ValueError('expected 2D or 3D input (got {}D input)'
        #                      .format(input.dim()))

    @staticmethod
    def expand(stat, *args, **kwargs):
        return stat


class BatchNorm2d(_BatchNorm):
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

# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/14 18:22

"""
class-wise estimation,
moving-average,
biased estimation,
bias-corrected,
half_maximums via total running_mean,
momentum decreases progressively,
.../(eps + half_max)
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN
from SampleRateLearning.stable_batchnorm import global_variables as batch_labels


class _BatchNorm(origin_BN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_classes=2):
        if not track_running_stats:
            raise NotImplementedError

        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        self.running_var = torch.zeros(num_features)
        self.eps = pow(self.eps, 0.5)

        self.num_classes = num_classes
        self.num_batches_tracked = torch.zeros(num_classes, dtype=torch.long)
        self.register_buffer('running_cls_means', torch.zeros(num_features,  num_classes))
        self.register_buffer('running_cls_maxes', torch.zeros(num_features, num_classes))

        self.relu = torch.nn.functional.relu

    def _check_input_dim(self, input):
        raise NotImplementedError

    @staticmethod
    def expand(stat, target_size):
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        self._check_input_dim(input)

        sz = input.size()
        if self.training:
            if input.dim() == 4:
                reduced_dim = (3, 2, 0)  # do not change the order
            elif input.dim() == 2:
                reduced_dim = (0, )
            else:
                raise NotImplementedError

            data = input.detach()
            if input.size(0) == batch_labels.batch_size:
                indices = batch_labels.indices
            else:
                indices = batch_labels.braid_indices

            if len(indices) != self.num_classes:
                raise ValueError

            cur_momentum = [1., ] * self.num_classes
            for c, group in enumerate(indices):
                if len(group) == 0:
                    continue
                self.num_batches_tracked[c] += 1
                samples = data[group]
                mean = torch.mean(samples, dim=reduced_dim, keepdim=False)
                cur_momentum[c] = self.momentum + (1. - self.momentum) ** self.num_batches_tracked[c]
                self.running_cls_means[:, c] = (1 - cur_momentum[c]) * self.running_cls_means[:, c] + cur_momentum[c] * mean

            # correction_factors = (1. - (1. - self.momentum) ** self.num_batches_tracked)
            # self.running_mean = (self.running_cls_means / correction_factors).mean(dim=1, keepdim=False)
            self.running_mean = self.running_cls_means.mean(dim=1, keepdim=False)
            data = data - self.expand(self.running_mean, sz)
            data = self.relu(data, inplace=True)

            for c, group in enumerate(indices):
                if len(group) == 0:
                    continue
                maxes = data[group]
                for dim in reduced_dim:
                    maxes, _ = maxes.max(dim, False)

                maxes /= 2.

                self.running_cls_maxes[:, c] = (1 - cur_momentum[c]) * self.running_cls_maxes[:, c] + cur_momentum[c] * maxes

            # Note: the running_var is running_max indeed, for convenience of external calling, it has not been renamed.
            # self.running_var, _ = (self.running_cls_maxes / correction_factors).max(dim=1, keepdim=False)
            self.running_var, _ = self.running_cls_maxes.max(dim=1, keepdim=False)

        # Note: the running_var is running_max indeed, for convenience of external calling, it has not been renamed.
        y = (input - self.expand(self.running_mean, sz)) \
            / self.expand(self.running_var + self.eps, sz)

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


# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/9/30 8:08

"""
MAPE is computed with running mean,
average MAPEs of all classes,
.../max(eps, MAPE),
bias-corrected
"""
import torch
from torch.nn.modules.batchnorm import _BatchNorm as origin_BN
from warnings import warn
from SampleRateLearning.stable_batchnorm import global_variables as batch_labels


class _BatchNorm(origin_BN):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.running_var = torch.zeros(num_features)
        self.eps = pow(self.eps, 0.5)
        self.relu = torch.nn.functional.relu

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
                reduced_dim = (0, )
            else:
                raise NotImplementedError

            self.num_batches_tracked += 1
            correction_factor = 1. - (1. - self.momentum) ** self.num_batches_tracked

            data = input.detach()
            if input.size(0) == batch_labels.batch_size:
                indices = batch_labels.indices
            else:
                indices = batch_labels.braid_indices

            means = []
            for group in indices:
                if len(group) == 0:
                    warn('There is no sample of at least one class in current batch, which is incompatible with SRL.')
                    continue
                samples = data[group]
                mean = torch.mean(samples, dim=reduced_dim, keepdim=False)
                means.append(mean)

            di_mean = sum(means) / len(means)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * di_mean

            MAPEs = []
            data = self.relu(data - self.expand(self.running_mean.detach()/correction_factor, sz))
            for group in indices:
                samples = data[group]
                MAPE = samples.mean(dim=reduced_dim, keepdim=False)
                MAPEs.append(MAPE)

            di_MAPE = sum(MAPEs) / len(MAPEs)
            # Note: the running_var is running_MAPE indeed, for convenience of external calling, it has not been renamed.
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * di_MAPE

        correction_factor = 1. - (1. - self.momentum) ** self.num_batches_tracked

        # Note: the running_var is running_MAPE indeed, for convenience of external calling, it has not been renamed.
        denominator = torch.full_like(self.running_var, self.eps).max(self.running_var / correction_factor)
        y = (input - self.expand(self.running_mean / correction_factor, sz)) \
            / self.expand(denominator, sz)

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


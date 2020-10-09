# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/9 13:11

"""
Convolution and fully connected modules which incorporate weight centralization
"""

from torch.nn import Conv2d as normal_conv2d
from torch.nn import Linear as normal_linear
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import Tensor
from warnings import warn


class Conv2d(normal_conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        if self.bias is not None:
            warn('A Conv2d layer has bias, which may be not suitable with weight centralization.')

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - weight_mean
        return self._conv_forward(input, weight)


class Linear(normal_linear):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        if self.bias is not None:
            warn('A Linear layer has bias, which may be not suitable with weight centralization.')

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        return F.linear(input, weight, self.bias)


def convert_model(module):
    if isinstance(module, DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = DataParallel(mod, device_ids=module.device_ids)
        return mod

    mod = module

    if isinstance(module, normal_conv2d):
        mod = Conv2d(module.in_channels,
                     module.out_channels,
                     module.kernel_size,
                     module.stride,
                     module.padding,
                     module.dilation,
                     module.groups,
                     module.bias is not None,
                     module.padding_mode)
        mod.weight = module.weight
        mod.bias = module.bias

    elif isinstance(module, normal_linear):
        mod = Linear(module.in_features,
                     module.out_features,
                     module.bias is not None)
        mod.weight = module.weight
        mod.bias = module.bias

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod
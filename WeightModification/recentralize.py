# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/9 19:48

"""
File Description

"""
from torch import nn


def recentralize(module: nn.Module):
    for model in module.modules():
        if isinstance(model, nn.Linear):
            weight = model.weight.data
            weight_means = weight.mean(dim=1, keepdim=True)
            model.weight.data -= weight_means

        elif isinstance(model, nn.Conv2d):
            weight = model.weight.data
            weight_means = weight.mean(dim=(1, 2, 3), keepdim=True)
            model.weight.data -= weight_means
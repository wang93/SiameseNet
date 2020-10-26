# encoding: utf-8
# author: Yicheng Wang
# contact: wyc@whu.edu.cn
# datetime:2020/10/24 23:56

"""
no relu activation

"""
import torch.nn as nn
import torch


class NoReLU(nn.Module):
    def __init__(self):
        super(NoReLU, self).__init__()

    def forward(self, input):
        return input


def convert_model(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = torch.nn.DataParallel(mod, device_ids=module.device_ids)
        return mod

    mod = module
    if isinstance(module, nn.ReLU):
        mod = NoReLU()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod

# coding=utf-8
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch.nn import BatchNorm3d, BatchNorm2d, BatchNorm1d
from torch.optim import SGD, Adam

from .subblocks import WConv2d, WBatchNorm2d, WLinear, WBatchNorm1d


def weights_init_kaiming(m: nn.Module):
    if isinstance(m, (nn.Linear, WLinear)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, WConv2d)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (BatchNorm1d, BatchNorm2d, BatchNorm3d, WBatchNorm1d, WBatchNorm2d)):
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    else:
        for _, n in m._modules.items():
            if n is None:
                continue
            weights_init_kaiming(n)


class BraidProto(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BraidProto, self).__init__()
        self.reg_params = []
        self.noreg_params = []

    def divide_params(self):
        self.check_pretrained_params()
        self.reg_params = []
        self.noreg_params = []
        for model in self.modules():
            for k, v in model._parameters.items():
                if v is None:
                    continue
                if k in ('weight',) and isinstance(model, (BatchNorm2d, BatchNorm1d, BatchNorm3d, WBatchNorm2d)):
                    self.noreg_params.append(v)
                else:
                    self.reg_params.append(v)

    def get_optimizer(self, optim='sgd', lr=0.1, momentum=0.9, weight_decay=0.0005):
        self.divide_params()

        if optim == "sgd":
            param_groups = [{'params': self.reg_params},
                            {'params': self.noreg_params, 'weight_decay': 0.}]
            default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
            optimizer = SGD(param_groups, **default)
        elif optim == 'adam':
            param_groups = [{'params': self.reg_params},
                            {'params': self.noreg_params, 'weight_decay': 0.}]
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             amsgrad=False)
        else:
            raise NotImplementedError

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        return optimizer

    def correct_params(self):
        for m in self.modules():
            if isinstance(m, (WConv2d, WLinear)):
                m.correct_params()

    def correct_grads(self):
        for m in self.modules():
            if isinstance(m, (WConv2d, WLinear)):
                m.correct_grads()

    @abstractmethod
    def extract(self, ims):
        pass

    @abstractmethod
    def metric(self, feat_a, feat_b):
        pass

    @abstractmethod
    def forward(self, a, b=None, mode='normal'):
        pass

    @abstractmethod
    def load_pretrained(self, *args, **kwargs):
        pass

    @abstractmethod
    def unlable_pretrained(self):
        pass

    @abstractmethod
    def check_pretrained_params(self):
        pass

    @abstractmethod
    def train(self, mode=True):
        pass

# coding=utf-8
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch.nn import BatchNorm3d, BatchNorm2d, BatchNorm1d



def weights_init_kaiming(m: nn.Module):
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,)):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
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

                if k in ('weight',) and isinstance(model, (nn.Conv2d, nn.Linear, nn.Conv3d)):
                    self.reg_params.append(v)

                else:
                    self.noreg_params.append(v)

                # if k in ('bias',):
                #     self.noreg_params.append(v)
                # elif k in ('weight',) and isinstance(model,
                #                                      (BatchNorm2d, BatchNorm1d, BatchNorm3d, WBatchNorm2d,
                #                                       WBatchNorm1d)):
                #     self.noreg_params.append(v)
                # else:
                #     self.reg_params.append(v)

    def get_optimizer(self, optim='sgd', lr=0.1, momentum=0.9, weight_decay=0.0005, gc=False, gc_loc=False):
        self.divide_params()
        if gc:
            if optim == "sgd":
                from WeightModification.optimizers import SGD
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
                optimizer = SGD(param_groups, **default, use_gc=gc)

            elif optim == "sgdw":
                raise NotImplementedError

            elif optim == 'adam':
                from WeightModification.optimizers import Adam
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'weight_decay': weight_decay}
                optimizer = Adam(param_groups, **default,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 amsgrad=False,
                                 use_gc=gc,
                                 gc_loc=gc_loc)

            elif optim == 'amsgrad':
                from WeightModification.optimizers import Adam
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'weight_decay': weight_decay}
                optimizer = Adam(param_groups, **default,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 amsgrad=True,
                                 use_gc=gc,
                                 gc_loc=gc_loc)

            elif optim == 'adamw':
                from WeightModification.optimizers import AdamW
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'weight_decay': weight_decay}
                optimizer = AdamW(param_groups, **default,
                                  betas=(0.9, 0.999),
                                  eps=1e-8,
                                  amsgrad=False,
                                  use_gc=gc,
                                  gc_loc=gc_loc)

        else:
            if optim == "sgd":
                from torch.optim import SGD
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
                optimizer = SGD(param_groups, **default)

            elif optim == "sgdw":
                from utils.optim.sgdw import SGDW
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
                optimizer = SGDW(param_groups, **default)

            elif optim == 'adam':
                from torch.optim import Adam
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'weight_decay': weight_decay}
                optimizer = Adam(param_groups, **default,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 amsgrad=False)

            elif optim == 'amsgrad':
                from torch.optim import Adam
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'weight_decay': weight_decay}
                optimizer = Adam(param_groups, **default,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 amsgrad=True)

            elif optim == 'adamw':
                from torch.optim import AdamW
                param_groups = [{'params': self.reg_params},
                                {'params': self.noreg_params, 'weight_decay': 0.}]
                default = {'lr': lr, 'weight_decay': weight_decay}
                optimizer = AdamW(param_groups, **default,
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
            if m is self:
                continue
            if hasattr(m, 'correct_params'):
                m.correct_params()

    def correct_grads(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, 'correct_grads'):
                m.correct_grads()

    def zero_tail_weight(self):
        nn.init.constant_(self.fc[-1].fc.weight, 0.0)

    @property
    def _default_output(self):
        return None

    @abstractmethod
    def extract(self, ims):
        pass

    @abstractmethod
    def metric(self, feat_a, feat_b):
        pass

    @abstractmethod
    def forward(self, a=None, b=None, mode='normal'):
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

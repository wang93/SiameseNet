# coding=utf-8
import torch.nn as nn
from optimizers import SGD2, Adam2
import torch.utils.model_zoo as model_zoo
from .blocks import Pair2Bi, BiBlock, Bi2Braid, BraidBlock, SumY, MaxY, SumMaxY, FCBlock
from .subblocks import WConv2d, WBatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class BraidNet(nn.Module):
    resnet2in = {
        'conv1.weight': 'bi.0.conv.weight',
        'bn1.weight': 'bi.0.bn.weight',
        'bn1.bias': 'bi.0.bn.bias',
        'bn1.running_mean': 'bi.0.bn.running_mean',
        'bn1.running_var': 'bi.0.bn.running_var',
    }

    reg_params = []
    noreg_params = []
    pretrained_params = []
    has_resnet_stem = False

    def __init__(self, bi, braid, fc):
        super(BraidNet, self).__init__()

        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [256, 128]
                     }

        channel_in = 3

        self.pair2bi = Pair2Bi()

        bi_blocks = []
        for i, sub_bi in enumerate(bi):
            kernel_size = 3 if i > 0 else 7
            stride = 1 if i > 0 else 2
            bi_blocks.append(BiBlock(channel_in, sub_bi, kernel_size=kernel_size, stride=stride))
            channel_in = sub_bi
        self.bi = nn.Sequential(*bi_blocks)

        self.bi2braid = Bi2Braid()

        braid_blocks = []
        for i, sub_braid in enumerate(braid):
            gap = (i+1 == len(braid))
            braid_blocks.append(BraidBlock(channel_in, sub_braid, gap=gap))
            channel_in = sub_braid
        self.braid = nn.Sequential(*braid_blocks)

        #self.y = SumY(channel_in)
        #self.y = MaxY(channel_in)
        self.y = SumMaxY(channel_in)
        channel_in *= 2

        self.fc_blocks = nn.ModuleList()
        for i, sub_fc in enumerate(fc):
            is_tail = (i+1 == len(fc))
            self.fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc

        self.score2prob = nn.Sigmoid()

        # initialize parameters
        for m in self.modules():
            weights_init_kaiming(m)

        self.correct_params()

    def forward(self, ims_a, ims_b):
        x = self.pair2bi(ims_a, ims_b)
        x = self.bi(x)
        x = self.bi2braid(x)
        x = self.braid(x)
        x = self.y(x)

        for fc in self.fc_blocks:
            x = fc(x)

        if self.training:
            return self.score2prob(x)
        else:
            return x

    def correct_params(self):
        for m in self.modules():
            if isinstance(m, WConv2d):
                m.correct_params()

    def correct_grads(self):
        for m in self.modules():
            if isinstance(m, WConv2d):
                m.correct_grads()

    def load_resnet_stem(self, resnet_name='resnet18'):
        resnet_state_dict = model_zoo.load_url(model_urls[resnet_name])

        in_state_dict = dict()
        for out_, in_ in self.resnet2in.items():
            in_state_dict[in_] = resnet_state_dict[out_]

        self.load_state_dict(in_state_dict, strict=False)
        self.has_resnet_stem = True

    def unlable_resnet_stem(self):
        self.has_resnet_stem = False

    def check_pretrained_params(self):
        self.pretrained_params = []
        if self.has_resnet_stem:
            for _, in_ in self.resnet2in.items():
                if '.running_' not in in_:
                    self.pretrained_params.append(self.get_indirect_attr(in_))

    def get_indirect_attr(self, name: str):
        attr = self
        for n in name.split('.'):
            attr = getattr(attr, n)

        return attr

    def divide_params(self):
        self.check_pretrained_params()
        self.reg_params = []
        self.noreg_params = []
        classified_params = set(self.pretrained_params)
        for model in self.modules():
            for k, v in model._parameters.items():
                if v is None or v in classified_params:
                    continue
                if k in ('weight', ) and isinstance(model, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, WBatchNorm2d)):
                    self.noreg_params.append(v)
                else:
                    self.reg_params.append(v)

    def get_optimizer(self, optim='sgd', lr=0.1, momentum=0.9, weight_decay=0.0005):
        self.divide_params()

        param_groups = [{'params': self.reg_params},
                        {'params': self.noreg_params, 'weight_decay': 0.},
                        {'params': self.pretrained_params, 'weight_decay': 0., 'base_lr': 0., 'lr': 0., 'momentum': 0.}]
        default = {'base_lr': lr, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}

        if optim == "sgd":
            optimizer = SGD2(param_groups, **default)
        else:
            optimizer = Adam2(param_groups, **default)

        return optimizer

    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        self.training = mode

        for name, module in self.named_children():
            module.train(mode)

        stem_training = mode and (not self.has_resnet_stem)
        self.bi[0].train(stem_training)

        return self

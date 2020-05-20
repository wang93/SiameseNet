# coding=utf-8
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.braidnet.primitives_v2.blocks import *
from .braidproto import BraidProto, weights_init_kaiming

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BraidNet(BraidProto):
    resnet2in = {
        'conv1.weight': 'bi.0.conv.weight',
        'bn1.weight': 'bi.0.bn.weight',
        'bn1.bias': 'bi.0.bn.bias',
        'bn1.running_mean': 'bi.0.bn.running_mean',
        'bn1.running_var': 'bi.0.bn.running_var',
    }

    reg_params = []
    noreg_params = []
    has_resnet_stem = False

    def __init__(self, bi, braid, fc, score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [128, 64]
                     }

        channel_in = 3

        self.pair2bi = Pair2Bi()

        self.pair2braid = Pair2Braid()

        bi_blocks = []
        for i, sub_bi in enumerate(bi):
            kernel_size = 3 if i > 0 else 7
            bi_blocks.append(BiBlock(channel_in, sub_bi, kernel_size=kernel_size, stride=1))
            channel_in = sub_bi
        self.bi = nn.Sequential(*bi_blocks)

        self.bi2braid = Bi2Braid()

        braid_blocks = []
        for i, sub_braid in enumerate(braid):
            gap = (i+1 == len(braid))
            braid_blocks.append(BraidBlock(channel_in, sub_braid, gap=gap))
            channel_in = sub_braid
        self.braid = nn.Sequential(*braid_blocks)

        self.y = MinMaxY(channel_in)
        channel_in *= 2

        fc_blocks = []
        for i, sub_fc in enumerate(fc):
            is_tail = (i+1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob  # nn.Sigmoid()

        # initialize parameters
        for m in self.modules():
            weights_init_kaiming(m)

        self.correct_params()

    def extract(self, ims):
        x = self.bi(ims)
        return x

    def metric(self, feat_a, feat_b):
        x = self.pair2braid(feat_a, feat_b)
        x = self.braid(x)
        x = self.y(x)
        x = self.fc(x)

        if self.training:
            return x
        else:
            return self.score2prob(x)

    def forward(self, a=None, b=None, mode='normal'):
        if a is None:
            return self._default_output
        if mode == 'extract':
            return self.extract(a)
        elif mode == 'metric':
            return self.metric(a, b)

        x = self.pair2bi(a, b)
        x = self.bi(x)
        x = self.bi2braid(x)
        x = self.braid(x)
        x = self.y(x)
        x = self.fc(x)

        if self.training:
            return x
        else:
            return self.score2prob(x)

    def load_pretrained(self, resnet_name='resnet18'):
        resnet_state_dict = model_zoo.load_url(model_urls[resnet_name])

        in_state_dict = dict()
        for out_, in_ in self.resnet2in.items():
            in_state_dict[in_] = resnet_state_dict[out_]

        self.load_state_dict(in_state_dict, strict=False)
        self.has_resnet_stem = True

    def unlable_pretrained(self):
        self.has_resnet_stem = False

    def check_pretrained_params(self):
        for param in self.bi[0].parameters():
            param.requires_grad = not self.has_resnet_stem

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

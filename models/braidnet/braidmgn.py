import copy

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

from .blocks import *
from .braidproto import BraidProto, weights_init_kaiming
from .subblocks import *


class MGN(nn.Module):
    def __init__(self, feats=256):
        super(MGN, self).__init__()

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.pool_zg_p1 = PartPool(part_num=1, method='avg')
        self.pool_zg_p2 = PartPool(part_num=1, method='avg')
        self.pool_zg_p3 = PartPool(part_num=1, method='avg')
        self.pool_zp2 = PartPool(part_num=2, method='avg')
        self.pool_zp3 = PartPool(part_num=3, method='avg')

        reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.pool_zg_p1(p1)
        zg_p2 = self.pool_zg_p2(p2)
        zg_p3 = self.pool_zg_p3(p3)

        zp2 = self.pool_zp2(p2)
        z0_p2, z1_p2 = torch.split(zp2, 1, dim=2)
        # z0_p2 = zp2[:, :, 0:1, :]
        # z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.pool_zp3(p3)
        z0_p3, z1_p3, z2_p3 = torch.split(zp3, 1, dim=2)
        # z0_p3 = zp3[:, :, 0:1, :]
        # z1_p3 = zp3[:, :, 1:2, :]
        # z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        fg = torch.cat((fg_p1, fg_p2, fg_p3), dim=1)

        return fg, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3


class BraidMGN(BraidProto):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,)):
        super(BraidMGN, self).__init__()

        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [384, 128]
                     }

        self.pair2bi = Pair2Bi()

        self.pair2braid = Pair2Braid()

        self.bi = MGN(feats=feats)

        self.bi2braid = Bi2Braid()

        channel_ins = [feats*3, feats, feats, feats, feats, feats]
        self.part_braids = nn.ModuleList()
        for channel_in in channel_ins:
            self.part_braids.append(LinearBraidBlock(channel_in, channel_in))

        self.braids2braid = CatBraids()
        channel_in = sum(channel_ins)

        self.final_braid = LinearBraidBlock(channel_in, channel_in)

        self.y = MinMaxY(channel_in, linear=True)
        channel_in *= 2

        fc_blocks = []
        for i, sub_fc in enumerate(fc):
            is_tail = (i+1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = nn.Sigmoid()

        # initialize parameters
        for m in [self.part_braids, self.final_braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()

    def load_pretrained(self, *args, **kwargs):
        pass

    def extract(self, ims):
        x = self.bi(ims)
        return x

    def metric(self, feat_a, feat_b):
        x = self.pair2braid(feat_a, feat_b)
        x = [model(data) for model, data in zip(self.part_braids, x)]
        x = self.braids2braid(x)
        x = self.final_braid(x)
        x = self.y(x)
        x = self.fc(x)

        if self.training:
            return self.score2prob(x)
        else:
            return x

    def forward(self, a=None, b=None, mode='normal'):
        if a is None:
            return None
        if mode == 'extract':
            return self.extract(a)
        elif mode == 'metric':
            return self.metric(a, b)

        x = self.pair2bi(a, b)
        x = self.bi(x)
        x = self.bi2braid(x)
        x = [model(data) for model, data in zip(self.part_braids, x)]
        x = self.braids2braid(x)
        x = self.final_braid(x)
        x = self.y(x)
        x = self.fc(x)

        if self.training:
            return self.score2prob(x)
        else:
            return x

    def unlable_pretrained(self):
        self.freeze_pretrained = False

    def check_pretrained_params(self):
        for model in [self.bi.backone, self.bi.p1, self.bi.p2, self.bi.p3]:
            for parameter in model.parameters():
                parameter.requires_grad = not self.freeze_pretrained

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

        pretrained_training = mode and (not self.freeze_pretrained)
        for model in [self.bi.backone, self.bi.p1, self.bi.p2, self.bi.p3]:
            model.train(pretrained_training)

        return self
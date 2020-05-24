# encoding: utf-8
from warnings import warn

import torch
import torch.nn as nn

from models.braidnet.primitives_v2.blocks import *
from models.osnet import osnet_x1_0, init_pretrained_weights
from .braidproto import BraidProto, weights_init_kaiming


class OSNet(BraidProto):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=512, num_classes=1000, **kwargs):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()

        self.bi = osnet_x1_0(feats=feats,
                             num_classes=num_classes)

        self.dist = nn.PairwiseDistance(p=2.0)  # nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_pretrained(self, *args, **kwargs):
        warn('some functions related to pretrained params have not been completed yet')
        init_pretrained_weights(self.bi, key='osnet_x1_0')

    def extract(self, ims):
        x = self.bi(ims)
        return x

    def metric(self, feat_a, feat_b):
        if self.training:
            raise NotImplementedError

        return - self.dist(feat_a, feat_b)

    def forward(self, a=None, b=None, mode='normal'):
        if a is None:
            return self._default_output
        if mode == 'extract':
            return self.extract(a)
        elif mode == 'metric':
            return self.metric(a, b)

        if self.training:
            raise NotImplementedError

        x = self.pair2bi(a, b)
        x = self.bi(x)
        a, b = torch.chunk(x, 2, dim=0)
        return - self.dist(a, b)

    def unlable_pretrained(self):
        pass

    def check_pretrained_params(self):
        pass

    def train(self, mode=True):
        torch.nn.Module.train(self, mode)


class EulideanOSNet(OSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=512, num_classes=1000, **kwargs):
        raise NotImplementedError('useless')
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()

        self.bi = osnet_x1_0(feats=feats,
                             num_classes=num_classes)

        self.cos = nn.PairwiseDistance(p=2.0)


class BraidOSNet(BraidProto):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=512, fc=(1,), score2prob=nn.Sigmoid(), **kwargs):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearBraidBlock(feats, feats)
        self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()

    def load_pretrained(self, *args, **kwargs):
        warn('some functions related to pretrained params have not been completed yet')
        init_pretrained_weights(self.bi, key='osnet_x1_0')

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

    def unlable_pretrained(self):
        pass

    def check_pretrained_params(self):
        pass

    def train(self, mode=True):
        torch.nn.Module.train(self, mode)


class MinMaxOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = nn.Identity()
        self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class MMBraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearMMBlock(feats, feats)
        self.y = MinMaxY(feats * 2, linear=True)

        fc_blocks = []
        channel_in = feats * 4
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class MinWMaxYBraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearMinBlock(feats, feats)
        self.y = MaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class MinWMMYBraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearMinBlock(feats, feats)
        self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class SquareOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = nn.Identity()
        self.y = SquareY(feats, linear=True)

        fc_blocks = []
        channel_in = feats
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class MeanSquareOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = nn.Identity()
        self.y = MeanSquareY(feats, linear=True)

        self.fc = nn.Identity()

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class SumSquareOSNet(MeanSquareOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = nn.Identity()
        self.y = SumSquareY(feats, linear=True)

        self.fc = nn.Identity()

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class AABraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = AABlock(feats, feats)
        # self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2 + feats
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()

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
        x = self.fc(x)

        if self.training:
            return x
        else:
            return self.score2prob(x)

    def metric(self, feat_a, feat_b):
        x = self.pair2braid(feat_a, feat_b)
        x = self.braid(x)
        x = self.fc(x)

        if self.training:
            return x
        else:
            return self.score2prob(x)


class AA2BraidOSNet(AABraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = AA2Block(feats, feats)
        # self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2 + feats
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class Min2WMMYBraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearMin2Block(feats, feats)
        self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class MinBNWMMYBraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearMinBNBlock(feats, feats)
        self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()


class MinBN2WMMYBraidOSNet(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, fc=(1,), score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats)
        self.bi.classifier = nn.Identity()
        self.bi2braid = Bi2Braid()
        self.braid = LinearMinBN2Block(feats, feats)
        self.y = MinMaxY(feats, linear=True)

        fc_blocks = []
        channel_in = feats * 2
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc
        self.fc = nn.Sequential(*fc_blocks)

        self.score2prob = score2prob

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()

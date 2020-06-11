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

        self.dist = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_pretrained(self, *args, **kwargs):
        warn('some functions related to pretrained params have not been completed yet')
        init_pretrained_weights(self.bi, key='osnet_x1_0')

    def extract(self, ims):
        if self.training:
            y, _ = self.bi(ims)
            return y
        else:
            v = self.bi(ims)
            return v

    def metric(self, feat_a, feat_b):
        if self.training:
            raise NotImplementedError

        return self.dist(feat_a, feat_b)

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
        return self.dist(a, b)

    def unlable_pretrained(self):
        pass

    def check_pretrained_params(self):
        pass

    def train(self, mode=True):
        torch.nn.Module.train(self, mode)


class BraidOSNet(BraidProto):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=512, fc=(1,), score2prob=nn.Sigmoid(), num_classes=1, no_classifier=True, **kwargs):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats, num_classes=num_classes)
        if no_classifier:
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
        if self.training:
            y, _ = self.bi(ims)
            return y
        else:
            v = self.bi(ims)
            return v

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


class BBOSNet(BraidOSNet):
    def __init__(self, feats=512, fc=(1,), score2prob=nn.Sigmoid(), num_classes=1, no_classifier=True, **kwargs):
        super(BBOSNet, self).__init__(feats=feats,
                                      fc=fc,
                                      score2prob=score2prob,
                                      num_classes=num_classes,
                                      no_classifier=no_classifier)

        self.fc_normal = FCBlock(feats, feats, is_tail=False)
        self.dist = nn.CosineSimilarity(dim=1, eps=1e-6)

        # initialize parameters
        for m in [self.fc_normal, ]:
            weights_init_kaiming(m)

    def metric(self, feat_a, feat_b):
        x = self.pair2braid(feat_a, feat_b)
        x = self.braid(x)
        x = self.y(x)
        x = self.fc(x)

        x += self.dist(self.fc_normal(feat_a), self.fc_normal(feat_b)).view(-1, 1)

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
        y = self.braid(x)
        y = self.y(y)
        y = self.fc(y)

        y += self.dist(self.fc_normal(x[0]), self.fc_normal(x[1])).view(-1, 1)

        if self.training:
            return y
        else:
            return self.score2prob(y)


class BBMOSNet(BBOSNet):
    def __init__(self, feats=512, fc=(1,), score2prob=nn.Sigmoid(), num_classes=1, no_classifier=True, **kwargs):
        super(BBMOSNet, self).__init__(feats=feats,
                                       fc=fc,
                                       score2prob=score2prob,
                                       num_classes=num_classes,
                                       no_classifier=no_classifier)
        self.braid = LinearMinBlock(feats, feats)

        # initialize parameters
        for m in [self.braid, ]:
            weights_init_kaiming(m)

        self.correct_params()


class WBBMOSNet(BBMOSNet):
    def __init__(self, feats=512, fc=(1,), score2prob=nn.Sigmoid(), num_classes=1, no_classifier=True, **kwargs):
        super(WBBMOSNet, self).__init__(feats=feats,
                                        fc=fc,
                                        score2prob=score2prob,
                                        num_classes=num_classes,
                                        no_classifier=no_classifier)

        self.weighted_sum = ADD()

    def metric(self, feat_a, feat_b):
        x = self.pair2braid(feat_a, feat_b)
        x = self.braid(x)
        x = self.y(x)
        y = self.fc(x)
        z = self.dist(self.fc_normal(feat_a), self.fc_normal(feat_b)).view(-1, 1)
        s = self.weighted_sum(y, z)

        if self.training:
            return s
        else:
            return self.score2prob(s)

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
        y = self.braid(x)
        y = self.y(y)
        y = self.fc(y)
        z = self.dist(self.fc_normal(x[0]), self.fc_normal(x[1])).view(-1, 1)
        s = self.weighted_sum(y, z)

        if self.training:
            return s
        else:
            return self.score2prob(s)


class WBBOSS(WBBMOSNet):
    def __init__(self, feats=512, fc=(1,), score2prob=nn.Sigmoid(), num_classes=1, **kwargs):
        super(WBBOSS, self).__init__(feats=feats,
                                     fc=fc,
                                     score2prob=score2prob,
                                     num_classes=num_classes,
                                     no_classifier=False)

        self.fc_normal = nn.Identity()

    def extract(self, ims):
        return self.bi(ims)


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

        # initialize parameters
        for m in [self.braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()

    def metric(self, feat_a, feat_b):
        x = self.pair2braid(feat_a, feat_b)
        x = self.braid(x)
        x = self.y(x)
        x = self.fc(x)

        if self.training:
            return -x
        else:
            return -x

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
            return -x
        else:
            return -x


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

    def half_forward(self, ims):
        """this method is used in checking discriminant"""
        x = self.bi(ims)
        x = self.braid.half_forward(x)
        return x

    def forward(self, a=None, b=None, mode='normal'):
        if a is None:
            return self._default_output
        if mode == 'extract':
            return self.extract(a)
        elif mode == 'half':
            return self.half_forward(a)
        elif mode == 'metric':
            return self.metric(a, b)
        elif mode == 'y':
            return self.get_y(a, b)
        elif mode == 'iy':
            return self.get_intermediate_y(a, b)

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

    def get_y(self, a, b):
        if self.training:
            raise AttributeError
        x = self.pair2braid(a, b)
        y = self.braid.get_y(x)
        return y

    def get_intermediate_y(self, a, b):
        if self.training:
            raise AttributeError
        x = self.pair2braid(a, b)
        return self.braid.get_intermediate_y(x)

    def get_y_effect(self):
        if self.training:
            raise AttributeError
        if len(self.fc) != 1:
            raise NotImplementedError
        weight = self.fc[0].fc.weight.view(-1)

        mask = self.braid.get_y_mask()

        return weight[mask]


class AAABraidOSNet(AABraidOSNet):
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
        self.braid = AAABlock(feats, feats)
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


class AAASBraidOSNet(AABraidOSNet):
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
        self.braid = AAASBlock(feats, feats)
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


class AABOSS(BraidOSNet):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=256, w_num=1, fc=(1,), num_classes=1000, score2prob=nn.Sigmoid()):
        nn.Module.__init__(self)

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()
        self.pair2braid = Pair2Braid()
        self.bi = osnet_x1_0(feats=feats, num_classes=num_classes)
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

    def half_forward(self, ims):
        """this method is used in checking discriminant"""
        if self.training:
            raise AttributeError
        x = self.bi(ims)
        x = self.braid.half_forward(x)
        return x

    def get_y_effect(self):
        if self.training:
            raise AttributeError
        if len(self.fc) != 1:
            raise NotImplementedError
        weight = self.fc[0].fc.weight.view(-1)
        mask = self.braid.get_y_mask()

        return weight[mask]

    def get_y(self, a, b):
        if self.training:
            raise AttributeError
        x = self.pair2braid(a, b)
        y = self.braid.get_y(x)
        return y

    def forward(self, a=None, b=None, mode='normal'):
        if a is None:
            return self._default_output
        if mode == 'extract':
            return self.extract(a)
        elif mode == 'half':
            return self.half_forward(a)
        elif mode == 'metric':
            return self.metric(a, b)
        elif mode == 'y':
            return self.get_y(a, b)

        raise NotImplementedError('phase_num==1 demands too complicated implementation')

    def extract(self, ims):
        return self.bi(ims)

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


class AA3BraidOSNet(AABraidOSNet):
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
        self.braid = AA3Block(feats, feats)
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


class AA4BraidOSNet(AABraidOSNet):
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
        self.braid = AA4Block(feats, feats)
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

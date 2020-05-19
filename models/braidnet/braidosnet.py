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
        super(OSNet, self).__init__()

        self.meta = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'imageSize': [256, 128]
        }

        self.pair2bi = Pair2Bi()

        self.bi = osnet_x1_0(feats=feats,
                             num_classes=num_classes)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_pretrained(self, *args, **kwargs):
        init_pretrained_weights(self.bi, key='osnet_x1_0')

    def extract(self, ims):
        x = self.bi(ims)
        return x

    def metric(self, feat_a, feat_b):
        if self.training:
            raise NotImplementedError

        return self.cos(feat_a, feat_b)

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
        return self.cos(a, b)


class BraidOSNet(BraidProto):
    reg_params = []
    noreg_params = []
    freeze_pretrained = True

    def __init__(self, feats=512, fc=(1,), num_classes=1000, pretrained=False, score2prob=nn.Sigmoid()):
        super(BraidProto, self).__init__()

        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [256, 128]
                     }

        self.pair2bi = Pair2Bi()

        self.pair2braid = Pair2Braid()

        self.bi = osnet_x1_0(feats=feats,
                             num_classes=num_classes)

        self.bi2braid = Bi2Braid()

        channel_in = feats
        self.final_braid = LinearBraidBlock(channel_in, channel_in)

        self.y = MinMaxY(channel_in, linear=True)
        channel_in *= 2

        fc_blocks = []
        for i, sub_fc in enumerate(fc):
            is_tail = (i + 1 == len(fc))
            fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc

        self.fc = nn.Sequential(*fc_blocks)
        self.score2prob = score2prob  # nn.Sigmoid()

        # initialize parameters
        for m in [self.part_braids, self.final_braid, self.fc]:
            weights_init_kaiming(m)

        self.correct_params()

    def load_pretrained(self, *args, **kwargs):
        init_pretrained_weights(self, key='osnet_x1_0')

    def extract(self, ims):
        x = self.bi(ims)
        return x

    def metric(self, feat_a, feat_b):
        if self.training:
            raise NotImplementedError

        x = self.pair2braid(feat_a, feat_b)
        x = [model(data) for model, data in zip(self.part_braids, x)]
        x = self.braids2braid(x)
        x = self.final_braid(x)
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
        x = [model(data) for model, data in zip(self.part_braids, x)]
        x = self.braids2braid(x)
        x = self.final_braid(x)
        x = self.y(x)
        x = self.fc(x)

        if self.training:
            return x
        else:
            return self.score2prob(x)

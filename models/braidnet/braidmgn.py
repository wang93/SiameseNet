import copy

from torchvision.models.resnet import resnet50, Bottleneck
from .subblocks import *

from optimizers import SGD2, Adam2
from .blocks import *  # Pair2Bi, BiBlock, Bi2Braid, BraidBlock, SumY, MaxY, SumMaxY, FCBlock
from .subblocks import WConv2d, WBatchNorm2d


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
        # nn.init.normal_(fc.weight, std=0.001)
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


class BraidMGN(nn.Module):

    reg_params = []
    noreg_params = []
    pretrained_params = []
    has_resnet_stem = False

    def __init__(self, feats=256, fc=(1,)):
        super(BraidMGN, self).__init__()

        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [384, 128]
                     }

        self.pair2bi = Pair2Bi()

        self.bi = MGN(feats=feats)

        self.bi2braid = Bi2Braid()

        channel_ins = [feats*3, feats, feats, feats, feats, feats]
        self.part_braids = nn.ModuleList()
        for channel_in in channel_ins:
            self.part_braids.append(LinearBraidBlock(channel_in, channel_in))

        self.braids2braid = CatBraids()
        channel_in = sum(channel_ins)
        self.final_braid = LinearBraidBlock(channel_in, channel_in)

        raise NotImplementedError

        #self.y = SumY(channel_in)
        #self.y = MaxY(channel_in)
        # self.y = SumMaxY(channel_in)
        # channel_in *= 2
        self.y = MinMaxY(channel_in)
        channel_in *= 2
        # self.y = SquareMaxY(channel_in)
        # self.y = ResMaxY(channel_in)
        # channel_in *= 2

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



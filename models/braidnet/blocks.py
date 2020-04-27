import torch.nn as nn
import torch
from .subblocks import WConv2d, WBatchNorm2d, WLinear, WBatchNorm1d


def int2tuple(n):
    if isinstance(n, int):
        n = (n, n)
    return n


class Pair2Bi(nn.Module):
    def forward(self, im_a, im_b):
        return torch.cat((im_a, im_b), dim=0)


class Pair2Braid(nn.Module):
    def forward(self, feat_a, feat_b):
        return torch.cat((feat_a, feat_b), dim=1)


class BiBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), stride=(1, 1)):
        super(BiBlock, self).__init__()
        kernel_size = int2tuple(kernel_size)
        stride = int2tuple(stride)
        padding = tuple([(i-1)//2 for i in kernel_size])
        self.conv = nn.Conv2d(channel_in, channel_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(channel_out,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=[3, 3],
                                 stride=[2, 2],
                                 padding=1,
                                 dilation=1,
                                 ceil_mode=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Bi2Braid(nn.Module):
    """transform bi-form features (maps/vectors) to braid-form features (maps/vectors)"""
    def __init__(self):
        super(Bi2Braid, self).__init__()
        self.transform = lambda x: torch.cat(torch.chunk(x, 2, dim=0), dim=1)

    def forward(self, x_from_bi):
        if isinstance(x_from_bi, torch.Tensor):
            return self.transform(x_from_bi)
        elif isinstance(x_from_bi, (list, tuple)):
            return [self.transform(x) for x in x_from_bi]
        else:
            raise NotImplementedError


class CatBraids(nn.Module):
    """cat the braid-form feature maps from multiple PartPool operations"""
    def __init__(self):
        super(CatBraids, self).__init__()

    def forward(self, braids):
        pairs = [torch.chunk(b, 2, dim=1) for b in braids]
        pairs = [*zip(*pairs)]
        parts = list(pairs[0]) + list(pairs[1])
        return torch.cat(parts, dim=1)


class CatBraidsGroups(nn.Module):
    """cat the braid-form feature maps from multiple PartPools operations"""
    def __init__(self):
        super(CatBraidsGroups, self).__init__()
        self.cat_braids = CatBraids()

    def forward(self, braids_groups):
        return [self.cat_braids(braids) for braids in zip(*braids_groups)]


class BraidBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), stride=(1, 1), gap=False):
        super(BraidBlock, self).__init__()
        kernel_size = int2tuple(kernel_size)
        stride = int2tuple(stride)
        padding = tuple([(i-1)//2 for i in kernel_size])
        self.wconv = WConv2d(channel_in, channel_out,
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=stride,
                             bias=False)
        self.wbn = WBatchNorm2d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        if gap:
            self.pool = nn.AdaptiveAvgPool2d(1)

        else:
            # self.pool = lambda x: x
            self.pool = nn.MaxPool2d(kernel_size=[2, 2],
                                     stride=[2, 2],
                                     padding=0,
                                     dilation=1,
                                     ceil_mode=False)

    def forward(self, x):
        x = self.wconv(x)
        x = self.wbn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class LinearBraidBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(LinearBraidBlock, self).__init__()
        self.wlinear = WLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.wlinear(x)
        x = self.wbn(x)
        x = self.relu(x)
        return x


class SumY(nn.Module):
    def __init__(self, channel_in, linear=False):
        super(SumY, self).__init__()
        if linear:
            self.bn = nn.BatchNorm1d(channel_in,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True,
                                     track_running_stats=True)
        else:
            self.bn = nn.BatchNorm2d(channel_in,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True,
                                     track_running_stats=True)

    def forward(self, x_from_braid):
        y = torch.add(*torch.chunk(x_from_braid, 2, dim=1))
        y = self.bn(y)
        return y.view(y.size(0), -1)


class MaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(MaxY, self).__init__(channel_in, linear)

    def forward(self, x_from_braid):
        y = torch.max(*torch.chunk(x_from_braid, 2, dim=1))
        y = self.bn(y)
        return y.view(y.size(0), -1)


class SumMaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(SumMaxY, self).__init__(channel_in*2, linear)

    def forward(self, x_from_braid):
        x = torch.chunk(x_from_braid, 2, dim=1)
        y_sum = torch.add(*x)
        y_max = torch.max(*x)
        y = torch.cat((y_sum, y_max), dim=1)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class MinMaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(MinMaxY, self).__init__(channel_in*2, linear)

    def forward(self, x_from_braid):
        x = torch.chunk(x_from_braid, 2, dim=1)
        y_min = torch.min(*x)
        y_max = torch.max(*x)
        y = torch.cat((y_min, y_max), dim=1)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class SquareMaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(SquareMaxY, self).__init__(channel_in*2, linear)

    def forward(self, x_from_braid):
        x = torch.chunk(x_from_braid, 2, dim=1)
        y_square = torch.sub(*x) ** 2
        y_max = torch.max(*x)
        y = torch.cat((y_square, y_max), dim=1)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class ResMaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(ResMaxY, self).__init__(channel_in*2, linear)

    def forward(self, x_from_braid):
        x = torch.chunk(x_from_braid, 2, dim=1)
        y_res = torch.abs(torch.sub(*x))
        y_max = torch.max(*x)
        y = torch.cat((y_res, y_max), dim=1)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class FCBlock(nn.Module):
    def __init__(self, channel_in, channel_out, is_tail=False):
        super(FCBlock, self).__init__()
        self.is_tail = is_tail
        self.fc = nn.Linear(channel_in, channel_out, bias=self.is_tail)
        if not self.is_tail:
            self.bn = nn.BatchNorm1d(channel_out,
                                     eps=1e-05,
                                     momentum=0.1,
                                     affine=True,
                                     track_running_stats=True)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if not self.is_tail:
            x = self.bn(x)
            x = self.relu(x)
        return x


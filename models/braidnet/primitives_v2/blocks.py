import torch
import torch.nn as nn
from torch.nn import BatchNorm1d as BatchNorm1d
from torch.nn import BatchNorm2d as BatchNorm2d

from utils.tensor_section_functions import cat_tensor_pair, combine_tensor_pair
from .subblocks import *

__all__ = ['BiBlock', 'Bi2Braid', 'Pair2Braid', 'Pair2Bi', 'CatBraids', 'LinearMin2Block', 'LinearMinBNBlock',
           'BraidBlock', 'LinearBraidBlock', 'SumY', 'MMBlock', 'LinearMMBlock', 'LinearMinBlock', 'AABlock',
           'AA2Block', 'SquareY', 'SumSquareY', 'MeanSquareY', 'AA3Block', 'AA4Block', 'AAABlock',
           'MinMaxY', 'FCBlock', 'DenseLinearBraidBlock', 'ResLinearBraidBlock', 'MaxY', 'MinY', 'LinearMinBN2Block']


def int2tuple(n):
    if isinstance(n, int):
        n = (n, n)
    return n


class BiBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), stride=(1, 1)):
        super(BiBlock, self).__init__()
        kernel_size = int2tuple(kernel_size)
        stride = int2tuple(stride)
        padding = tuple([(i - 1) // 2 for i in kernel_size])
        self.conv = nn.Conv2d(channel_in, channel_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)
        self.bn = BatchNorm2d(channel_out,
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
        self.transform = lambda x: torch.chunk(x, 2, dim=0)

    def forward(self, x_from_bi):
        if isinstance(x_from_bi, torch.Tensor):
            return self.transform(x_from_bi)
        elif isinstance(x_from_bi, (list, tuple)):
            return [self.transform(x) for x in x_from_bi]
        else:
            raise NotImplementedError


class Pair2Bi(nn.Module):
    def forward(self, im_a, im_b):
        return cat_tensor_pair(im_a, im_b, dim=0)


class Pair2Braid(nn.Module):
    def forward(self, feat_a, feat_b):
        return combine_tensor_pair(feat_a, feat_b)


class CatBraids(nn.Module):
    """cat the braid-form feature maps from multiple PartPool operations"""

    def forward(self, braids):
        # pairs = [torch.chunk(b, 2, dim=1) for b in braids]
        pair = [*zip(*braids)]
        pair = [torch.cat(e, dim=1) for e in pair]
        return pair


# class CatBraidsGroups(nn.Module):
#     """cat the braid-form feature maps from multiple PartPools operations"""
#     def __init__(self):
#         super(CatBraidsGroups, self).__init__()
#         self.cat_braids = CatBraids()
#
#     def forward(self, braids_groups):
#         return [self.cat_braids(braids) for braids in zip(*braids_groups)]


class BraidBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), stride=(1, 1), gap=False):
        super(BraidBlock, self).__init__()
        kernel_size = int2tuple(kernel_size)
        stride = int2tuple(stride)
        padding = tuple([(i - 1) // 2 for i in kernel_size])
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
        x = [self.relu(i) for i in x]
        x = [self.pool(i) for i in x]
        return x


class MMBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), stride=(1, 1), gap=False):
        # if channel_out % 2:
        #     raise ValueError
        super(MMBlock, self).__init__()

        kernel_size = int2tuple(kernel_size)
        stride = int2tuple(stride)
        padding = tuple([(i - 1) // 2 for i in kernel_size])
        self.wconv = MMConv2d(channel_in, channel_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)
        self.wbn = WBatchNorm2d(channel_out * 2,
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
        x = [self.relu(i) for i in x]
        x = [self.pool(i) for i in x]
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
        x = [self.relu(i) for i in x]
        return x


class LinearMMBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        # if channel_out % 2:
        #     raise ValueError
        super(LinearMMBlock, self).__init__()

        self.wlinear = MMLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out * 2,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.wlinear(x)
        x = self.wbn(x)
        x = [self.relu(i) for i in x]
        return x


class LinearMinBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        # if channel_out % 2:
        #     raise ValueError
        super(LinearMinBlock, self).__init__()

        self.wlinear = MinLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.wlinear(x)
        x = self.wbn(x)
        x = [self.relu(i) for i in x]
        return x


class LinearMinBN2Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(LinearMinBN2Block, self).__init__()

        self.wlinear = MinBNLinear(channel_in, channel_out)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.wlinear(x)
        x = self.wbn(x)
        x = [self.relu(i) for i in x]
        return x


class LinearMinBNBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(LinearMinBNBlock, self).__init__()

        self.wlinear = MinBNLinear(channel_in, channel_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.wlinear(x)
        x = [self.relu(i) for i in x]
        return x


class LinearMin2Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        # if channel_out % 2:
        #     raise ValueError
        super(LinearMin2Block, self).__init__()

        self.wlinear = Min2Linear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.wlinear(x)
        x = self.wbn(x)
        x = [self.relu(i) for i in x]
        return x


class DenseLinearBraidBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DenseLinearBraidBlock, self).__init__()
        self.wlinear = WLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.cat = CatBraids()

    def forward(self, x):
        y = self.wlinear(x)
        y = self.wbn(y)
        y = [self.relu(i) for i in y]
        z = self.cat([x, y])
        return z


class ResLinearBraidBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResLinearBraidBlock, self).__init__()
        if channel_in != channel_out:
            raise NotImplementedError

        self.wlinear = WLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.wbn2 = WBatchNorm1d(channel_out,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True)

    def forward(self, x):
        y = self.wlinear(x)
        y = self.wbn(y)
        y = [self.relu(i) for i in y]
        z = [i + j for i, j in zip(x, y)]
        z = self.wbn2(z)

        return z


class SumY(nn.Module):
    def __init__(self, channel_in, linear=False):
        super(SumY, self).__init__()
        if linear:
            self.bn = BatchNorm1d(channel_in,
                                  eps=1e-05,
                                  momentum=0.1,
                                  affine=True,
                                  track_running_stats=True)
        else:
            self.bn = BatchNorm2d(channel_in,
                                  eps=1e-05,
                                  momentum=0.1,
                                  affine=True,
                                  track_running_stats=True)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        y = torch.add(*x_from_braid)
        y = self.bn(y)
        return y.view(y.size(0), -1)


# class SumMaxY(SumY):
#     def __init__(self, channel_in, linear=False):
#         super(SumMaxY, self).__init__(channel_in * 2, linear)
#
#     def forward(self, x_from_braid):
#         x = torch.chunk(x_from_braid, 2, dim=1)
#         y_sum = torch.add(*x)
#         y_max = torch.max(*x)
#         y = torch.cat((y_sum, y_max), dim=1)
#         y = self.bn(y)
#         return y.view(y.size(0), -1)
#
#
# class SumMinY(SumY):
#     def __init__(self, channel_in, linear=False):
#         super(SumMinY, self).__init__(channel_in*2, linear)
#
#     def forward(self, x_from_braid):
#         x = torch.chunk(x_from_braid, 2, dim=1)
#         y_sum = torch.add(*x)
#         y_min = torch.min(*x)
#         y = torch.cat((y_sum, y_min), dim=1)
#         y = self.bn(y)
#         return y.view(y.size(0), -1)
class MaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(MaxY, self).__init__(channel_in, linear)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        y = torch.max(*x_from_braid)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class SquareY(SumY):
    def __init__(self, channel_in, linear=False):
        super(SquareY, self).__init__(channel_in, linear)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        y = torch.sub(*x_from_braid).pow(2.)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class SumSquareY(nn.Module):
    def __init__(self, channel_in, linear=False):
        nn.Module.__init__(self)
        # super(SumSquareY, self).__init__(channel_in, linear)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        y = torch.sub(*x_from_braid).pow(2.)
        y = torch.sum(y, dim=1)
        return y.view(y.size(0), -1)


class MeanSquareY(nn.Module):
    def __init__(self, channel_in, linear=False):
        nn.Module.__init__(self)
        if not linear:
            raise NotImplementedError
        # super(SumSquareY, self).__init__(channel_in, linear)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        y = torch.sub(*x_from_braid).pow(2.)
        y = torch.mean(y, dim=1)
        return y.view(y.size(0), -1)


class MinY(SumY):
    def __init__(self, channel_in, linear=False):
        super(MinY, self).__init__(channel_in, linear)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        y = torch.min(*x_from_braid)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class MinMaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(MinMaxY, self).__init__(channel_in * 2, linear)

    def forward(self, x_from_braid):
        if len(x_from_braid) != 2:
            raise NotImplementedError
        x = x_from_braid
        y_min = torch.min(*x)
        y_max = torch.max(*x)
        y = torch.cat((y_min, y_max), dim=1)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class SquareMaxY(SumY):
    def __init__(self, channel_in, linear=False):
        super(SquareMaxY, self).__init__(channel_in * 2, linear)

    def forward(self, x_from_braid):
        y_square = torch.sub(*x_from_braid).pow(2.)
        y_max = torch.max(*x_from_braid)
        y = torch.cat((y_square, y_max), dim=1)
        y = self.bn(y)
        return y.view(y.size(0), -1)


class AABlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AABlock, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.wlinear = MinLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.max_y = MaxY(channel_out, linear=True)
        self.min_max_y = MinMaxY(channel_in, linear=True)

    def forward(self, x):
        y = self.wlinear(x)
        y = self.wbn(y)
        y = [self.relu(i) for i in y]
        y = self.max_y(y)
        z = self.min_max_y(x)
        out = torch.cat((y, z), dim=1)
        return out

    def get_y(self, x):
        y = self.wlinear(x)
        y = self.wbn(y)
        # y = [self.relu(i) for i in y]
        y = self.max_y(y)
        return y

    def get_y_mask(self):
        mask = [i for i in range(self.channel_out)]
        return mask

    def half_forward(self, x):
        """this method is used in checking discriminant"""
        raise NotImplementedError
        y = self.wlinear.half_forward(x)
        return torch.cat((x, y), dim=1)


class AAABlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AAABlock, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.wlinear = AndLinear(channel_in, channel_out, bias=False)
        # self.wbn = WBatchNorm1d(channel_out,
        #                         eps=1e-05,
        #                         momentum=0.1,
        #                         affine=True,
        #                         track_running_stats=True)
        # self.relu = nn.ReLU(inplace=True)
        self.max_y = MaxY(channel_out, linear=True)
        self.min_max_y = MinMaxY(channel_in, linear=True)

    def forward(self, x):
        y = self.wlinear(x)
        # y = self.wbn(y)
        # y = [self.relu(i) for i in y]
        y = self.max_y(y)
        z = self.min_max_y(x)
        out = torch.cat((y, z), dim=1)
        return out

    def get_y(self, x):
        y = self.wlinear(x)
        # y = self.wbn(y)
        # y = [self.relu(i) for i in y]
        y = self.max_y(y)
        return y

    def get_y_mask(self):
        mask = [i for i in range(self.channel_out)]
        return mask

    def half_forward(self, x):
        """this method is used in checking discriminant"""
        raise NotImplementedError
        y = self.wlinear.half_forward(x)
        return torch.cat((x, y), dim=1)


class W2AABlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(W2AABlock, self).__init__()
        self.wblocks = nn.Sequential(MinLinear(channel_in, channel_out, bias=False),
                                     WBatchNorm1d(channel_out,
                                                  eps=1e-05,
                                                  momentum=0.1,
                                                  affine=True,
                                                  track_running_stats=True),
                                     ReLU(inplace=True),
                                     MinLinear(channel_out, channel_out, bias=False),
                                     WBatchNorm1d(channel_out,
                                                  eps=1e-05,
                                                  momentum=0.1,
                                                  affine=True,
                                                  track_running_stats=True),
                                     ReLU(inplace=True))

        self.max_y = MaxY(channel_out, linear=True)
        self.min_max_y = MinMaxY(channel_in, linear=True)

    def forward(self, x):
        y = self.wblocks(x)
        y = self.max_y(y)
        z = self.min_max_y(x)
        out = torch.cat((y, z), dim=1)
        return out

    def half_forward(self, x):
        """this method is used in checking discriminant"""
        y = self.wlinear.half_forward(x)
        return torch.cat((x, y), dim=1)


class AA3Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AA3Block, self).__init__()
        self.wlinear = MinLinear(channel_in, channel_out, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.max_y = MaxY(channel_out, linear=True)
        self.min_max_y = MinMaxY(channel_in, linear=True)

    def forward(self, x):
        y = self.wlinear(x)
        y = [self.relu(i) for i in y]
        y = self.max_y(y)
        z = self.min_max_y(x)
        out = torch.cat((y, z), dim=1)
        return out

    def half_forward(self, x):
        """this method is used in checking discriminant"""
        y = self.wlinear.half_forward(x)
        return torch.cat((x, y), dim=1)


class AA2Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AA2Block, self).__init__()
        self.wlinear = MinLinear(channel_in, channel_out, bias=False)
        self.wbn = WBatchNorm1d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.max_y = MaxY(channel_out, linear=True)
        self.square_max_y = SquareMaxY(channel_in, linear=True)

    def forward(self, x):
        y = self.wlinear(x)
        y = self.wbn(y)
        y = [self.relu(i) for i in y]
        y = self.max_y(y)
        z = self.square_max_y(x)
        out = torch.cat((y, z), dim=1)
        return out


class AA4Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(AA4Block, self).__init__()
        self.wlinear = MinLinear(channel_in, channel_out, bias=False)
        self.cs = ChanelScaling(channel_out, linear=True)
        self.relu = nn.ReLU(inplace=True)
        self.max_y = MaxY(channel_out, linear=True)
        self.square_max_y = SquareMaxY(channel_in, linear=True)

    def forward(self, x):
        y = self.wlinear(x)
        y = [self.cs(i) for i in y]
        y = [self.relu(i) for i in y]
        y = self.max_y(y)
        z = self.square_max_y(x)
        out = torch.cat((y, z), dim=1)
        return out


#
# class ResMaxY(SumY):
#     def __init__(self, channel_in, linear=False):
#         super(ResMaxY, self).__init__(channel_in*2, linear)
#
#     def forward(self, x_from_braid):
#         x = torch.chunk(x_from_braid, 2, dim=1)
#         y_res = torch.abs(torch.sub(*x))
#         y_max = torch.max(*x)
#         y = torch.cat((y_res, y_max), dim=1)
#         y = self.bn(y)
#         return y.view(y.size(0), -1)


class FCBlock(nn.Module):
    def __init__(self, channel_in, channel_out, is_tail=False):
        super(FCBlock, self).__init__()
        self.is_tail = is_tail
        self.fc = nn.Linear(channel_in, channel_out, bias=self.is_tail)
        if not self.is_tail:
            self.bn = BatchNorm1d(channel_out,
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

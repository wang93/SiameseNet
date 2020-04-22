import torch.nn as nn
import torch
from .subblocks import WConv2d, WBatchNorm2d


def int2tuple(n):
    if isinstance(n, int):
        n = (n, n)
    return n


class Pair2Bi(nn.Module):
    def __init__(self):
        super(Pair2Bi, self).__init__()

    def forward(self, im_a, im_b):
        return torch.cat((im_a, im_b), dim=0)


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
        # self.pool = nn.MaxPool2d(kernel_size=[2, 2],
        #                          stride=[2, 2],
        #                          padding=0,
        #                          dilation=1,
        #                          ceil_mode=False)
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
    def __init__(self):
        super(Bi2Braid, self).__init__()

    def forward(self, x_from_bi):
        return torch.cat(torch.chunk(x_from_bi, 2, dim=0), dim=1)


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


class SumY(nn.Module):
    def __init__(self, channel_in):
        super(SumY, self).__init__()
        self.bn = nn.BatchNorm2d(channel_in,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True)

    def forward(self, x_from_braid):
        y = torch.add(*torch.chunk(x_from_braid, 2, dim=1))
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


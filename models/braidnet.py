# coding=utf-8
import torch.nn as nn
import torch
#import abc
#from torch._jit_internal import weak_module, weak_script_method
from torch.nn import functional as F
#from torch.nn.parameter import Parameter
from optimizers import SGD2, Adam2
import torch.utils.model_zoo as model_zoo
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
            #nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# class _BraidModule(metaclass=abc.ABCMeta):
#     @abc.abstractmethod
#     def correct_params(self):
#         pass
#
#     @abc.abstractmethod
#     def correct_grads(self):
#         pass
def int2tuple(n):
    if isinstance(n, int):
        n = (n, n)

    return n


class WConv2d(nn.Conv2d):
    def __init__(self, in_channels=10, out_channels=10, kernel_size=3, stride=(1, 1),
                 padding=(1, 1), dilation=1, groups=1, bias=True):
        if groups != 1:
            raise NotImplementedError

        nn.Conv2d.__init__(
            self, 2 * in_channels, 2 * out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

        self.correct_params()

    def correct_params(self):
        weight_a = self.weight.data[:self.out_channels//2, :, :, :]
        p, q = torch.chunk(weight_a, 2, dim=1)
        weight_b = torch.cat((q, p), dim=1)
        weight_corrected = torch.cat((weight_a, weight_b), dim=0)
        self.weight.data = weight_corrected.data

        if self.bias is not None:
            bias_a = self.bias.data[:self.out_channels//2]
            bias_b = bias_a
            bias_corrected = torch.cat((bias_a, bias_b), dim=0)
            self.bias.data = bias_corrected.data

    def correct_grads(self):
        grad_a, grad_b = torch.chunk(self.weight.grad, 2, dim=0)
        q, p = torch.chunk(grad_b, 2, dim=1)
        grad_a = grad_a + torch.cat((p, q), dim=1)

        p, q = torch.chunk(grad_a, 2, dim=1)
        grad_b = torch.cat((q, p), dim=1)

        grad_corrected = torch.cat((grad_a, grad_b), dim=0)
        self.weight.grad.data = grad_corrected.data

        if self.bias is not None:
            grad_a, grad_b = torch.chunk(self.bias.grad, 2, dim=0)
            grad_a = grad_a + grad_b
            grad_b = grad_a
            grad_corrected = torch.cat((grad_a, grad_b), dim=0)
            self.bias.grad.data = grad_corrected.data

    def extra_repr(self):
        s = 'in_channels={0}, out_channels={1}'.format(int(self.__dict__['in_channels']/2), int(self.__dict__['out_channels']/2))
        s += ', kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class WBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_channels, eps=1e-5, **kwargs):
        nn.BatchNorm2d.__init__(self, num_features=num_channels, eps=eps, **kwargs)

    def forward(self, input):
        self._check_input_dim(input)

        # if self.training:
        input = torch.cat(torch.chunk(input, 2, dim=1), dim=0)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        output = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        output = torch.cat(torch.chunk(output, 2, dim=0), dim=1)
        # else:
        #
        #     output = F.batch_norm(
        #         input, self.running_mean.repeat(2), self.running_var.repeat(2), self.weight.repeat(2), self.bias.repeat(2),
        #         not self.track_running_stats,
        #         0.0, self.eps)

        return output

    def extra_repr(self):
        return 'num_channels={num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(num_features=int(self.num_features/2),
                                                                  eps=self.eps,
                                                                  momentum=self.momentum,
                                                                  affine=self.affine,
                                                                  track_running_stats=self.track_running_stats)


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


class BraidNet(nn.Module):
    resnet2in = {
        'conv1.weight': 'bi.0.conv.weight',
        'bn1.weight': 'bi.0.bn.weight',
        'bn1.bias': 'bi.0.bn.bias',
        'bn1.running_mean': 'bi.0.bn.running_mean',
        'bn1.running_var': 'bi.0.bn.running_var',
    }

    def __init__(self, bi, braid, fc):
        super(BraidNet, self).__init__()
        # self.meta = {'mean': [0.3578, 0.3544, 0.3471],
        #              'std': [1, 1, 1],
        #              'imageSize': [256, 128]
        #              }
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
            #stride = (1, 1) if gap else (2, 2)
            #braid_blocks.append(BraidBlock(channel_in, sub_braid, stride=stride, gap=gap))
            braid_blocks.append(BraidBlock(channel_in, sub_braid, gap=gap))
            channel_in = sub_braid
        self.braid = nn.Sequential(*braid_blocks)

        self.sumy = SumY(channel_in)

        self.fc_blocks = nn.ModuleList()
        for i, sub_fc in enumerate(fc):
            is_tail = (i+1 == len(fc))
            self.fc_blocks.append(FCBlock(channel_in, sub_fc, is_tail=is_tail))
            channel_in = sub_fc

        self.score2prob = nn.Sigmoid()

        #initialize parameters
        for m in self.modules():
            weights_init_kaiming(m)

        self.correct_params()
        self.pretrained_params = []
        self.has_resnet_stem = False#Parameter(torch.tensor(False), requires_grad=False)

    def forward(self, ims_a, ims_b):
        x = self.pair2bi(ims_a, ims_b)

        # for bi in self.bi_blocks:
        #     x = bi(x)
        x = self.bi(x)

        x = self.bi2braid(x)

        # for braid in self.braid_blocks:
        #     x = braid(x)
        x = self.braid(x)

        x = self.sumy(x)

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
        self.has_resnet_stem = True #torch.tensor(True)

    def unlable_resnet_stem(self):
        self.has_resnet_stem = False #torch.tensor(False)

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
        # print('braidnet has {0} params'.format(len(list(self.parameters()))))
        # print('braidnet has {0} reg_params'.format(len(self.reg_params)))
        # print('braidnet has {0} noreg_params'.format(len(self.noreg_params)))
        # print('braidnet has {0} pretrained_params'.format(len(self.pretrained_params)))

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
        #print('stem.training is {0}'.format(stem_training))
        self.bi[0].train(stem_training)

        return self

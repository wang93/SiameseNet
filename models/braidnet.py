# coding=utf-8
import torch.nn as nn
import torch
import abc
#from torch._jit_internal import weak_module, weak_script_method
from torch.nn import functional as F
from torch.nn.parameter import Parameter

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

# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         nn.init.normal_(m.weight, std=0.001)
#         if m.bias:
#             nn.init.constant_(m.bias, 0.0)


class _BraidModule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def correct_params(self):
        pass

    @abc.abstractmethod
    def correct_grads(self):
        pass


class WConv2d(nn.Conv2d, _BraidModule):
    def __init__(self, in_channels=10, out_channels=10, kernel_size=3, stride=(1, 1),
                 padding=(1, 1), dilation=1, groups=1, bias=True):
        if groups != 1:
            raise NotImplementedError

        nn.Conv2d.__init__(
            self, 2 * in_channels, 2 * out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

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

    # def extra_repr(self):
    #     s = 'in_channels={0}, out_channels={1}'.format(int(self.__dict__['in_channels']/2), int(self.__dict__['out_channels']/2))
    #     s += ', kernel_size={kernel_size}, stride={stride}'
    #     if self.padding != (0,) * len(self.padding):
    #         s += ', padding={padding}'
    #     if self.dilation != (1,) * len(self.dilation):
    #         s += ', dilation={dilation}'
    #     if self.output_padding != (0,) * len(self.output_padding):
    #         s += ', output_padding={output_padding}'
    #     if self.groups != 1:
    #         s += ', groups={groups}'
    #     if self.bias is None:
    #         s += ', bias=False'
    #     return s.format(**self.__dict__)


#@weak_module
class WBatchNorm2d(nn.BatchNorm2d, _BraidModule):
    def __init__(self, num_channels, eps=1e-5, **kwargs):
        #nn.BatchNorm2d.__init__(self, num_features=2*num_channels, eps=eps, **kwargs)
        nn.BatchNorm2d.__init__(self, num_features=num_channels, eps=eps, **kwargs)
        self._set_eval_params()

    def _set_eval_params(self):
        self.eval_running_mean = Parameter(torch.cat((self.running_mean.detach(), self.running_mean.detach()), dim=0))
        self.eval_running_var = Parameter(torch.cat((self.running_var.detach(), self.running_var.detach()), dim=0))
        self.eval_weight = Parameter(torch.cat((self.weight.detach(), self.weight.detach()), dim=0))
        self.eval_bias = Parameter(torch.cat((self.bias.detach(), self.bias.detach()), dim=0))

    def train(self, mode=True):
        r"""Sets the module in training mode.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        if not mode:
            self._set_eval_params()

        return self

#    @weak_script_method
    def forward(self, input):
        self._check_input_dim(input)

        if self.training:
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
        else:
            exponential_average_factor = 0.0

            output = F.batch_norm(
                input, self.eval_running_mean, self.eval_running_var, self.eval_weight, self.eval_bias,
                not self.track_running_stats,
                exponential_average_factor, self.eps)

        return output

    def correct_params(self):
        pass

    def correct_grads(self):
        if self.affine:
            self.weight.grad.mul_(2.)
            self.bias.grad.mul_(2.)

    # def extra_repr(self):
    #     return 'num_channels={num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
    #            'track_running_stats={track_running_stats}'.format(num_features=int(self.__dict__['num_features']/2),
    #                                                               eps=self.__dict__['eps'],
    #                                                               momentum=self.__dict__['momentum'],
    #                                                               affine=self.__dict__['affine'],
    #                                                               track_running_stats=self.__dict__['track_running_stats'])


class Pair2Bi(nn.Module):
    def __init__(self):
        super(Pair2Bi, self).__init__()

    def forward(self, im_a, im_b):
        return torch.cat((im_a, im_b), dim=0)


class BiBlock(nn.Module, _BraidModule):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3)):
        super(BiBlock, self).__init__()
        padding = tuple([(i-1)//2 for i in kernel_size])
        self.conv = nn.Conv2d(channel_in, channel_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=(1, 1),
                              bias=False)
        self.bn = nn.BatchNorm2d(channel_out,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=[2, 2],
                                 stride=[2, 2],
                                 padding=0,
                                 dilation=1,
                                 ceil_mode=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    def correct_params(self):
        pass

    def correct_grads(self):
        for p in self.parameters():
            p.grad.mul_(2.)


class Bi2Braid(nn.Module):
    def __init__(self):
        super(Bi2Braid, self).__init__()

    def forward(self, x_from_bi):
        return torch.cat(torch.chunk(x_from_bi, 2, dim=0), dim=1)


class BraidBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), gap=False):
        super(BraidBlock, self).__init__()
        padding = tuple([(i-1)//2 for i in kernel_size])
        self.wconv = WConv2d(channel_in, channel_out,
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=(1, 1),
                             bias=False)
        self.wbn = WBatchNorm2d(channel_out,
                                eps=1e-05,
                                momentum=0.1,
                                affine=True,
                                track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        if not gap:
            self.pool = nn.MaxPool2d(kernel_size=[2, 2],
                                     stride=[2, 2],
                                     padding=0,
                                     dilation=1,
                                     ceil_mode=False)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

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
    def __init__(self, bi, braid, fc):
        super(BraidNet, self).__init__()
        self.meta = {'mean': [0.3578, 0.3544, 0.3471],
                     'std': [1, 1, 1],
                     'imageSize': [128, 64]
                     }

        channel_in = 3

        self.pair2bi = Pair2Bi()

        self.bi_blocks = nn.ModuleList()
        for sub_bi in bi:
            self.bi_blocks.append(BiBlock(channel_in, sub_bi))
            channel_in = sub_bi

        self.bi2braid = Bi2Braid()

        self.braid_blocks = nn.ModuleList()
        for i, sub_braid in enumerate(braid):
            gap = (i+1 == len(braid))
            self.braid_blocks.append(BraidBlock(channel_in, sub_braid, gap=gap))
            channel_in = sub_braid

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

        # self.register_backward_hook(self.hook_correct_grads)

    def forward(self, ims_a, ims_b):
        x = self.pair2bi(ims_a, ims_b)

        for bi in self.bi_blocks:
            x = bi(x)

        x = self.bi2braid(x)

        for braid in self.braid_blocks:
            x = braid(x)

        x = self.sumy(x)

        for fc in self.fc_blocks:
            x = fc(x)

        if self.training:
            return self.score2prob(x)
        else:
            return x

    def correct_params(self):
        for m in self.modules():
            if isinstance(m, _BraidModule):
                m.correct_params()

    def correct_grads(self):
        for m in self.modules():
            if isinstance(m, _BraidModule):
                m.correct_grads()

    @staticmethod
    def hook_correct_grads(module, *args, **kwargs):
        module.correct_grads()

    def get_optim_policy(self):
        return self.parameters()

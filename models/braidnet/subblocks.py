import torch.nn as nn
import torch
from torch.nn import functional as F


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


class WLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        nn.Linear.__init__(self, 2 * in_features, 2 * out_features, bias)
        self.correct_params()

    def correct_params(self):
        weight_a = self.weight.data[:self.out_channels//2, :]
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
        raise NotImplementedError


class WBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_channels, eps=1e-5, **kwargs):
        nn.BatchNorm2d.__init__(self, num_features=num_channels, eps=eps, **kwargs)

    def forward(self, input_):
        self._check_input_dim(input_)

        input_ = torch.cat(torch.chunk(input_, 2, dim=1), dim=0)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        output = F.batch_norm(
            input_, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        output = torch.cat(torch.chunk(output, 2, dim=0), dim=1)

        return output


class WBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_channels, eps=1e-5, **kwargs):
        nn.BatchNorm1d.__init__(self, num_features=num_channels, eps=eps, **kwargs)

    def forward(self, input_):
        self._check_input_dim(input_)

        input_ = torch.cat(torch.chunk(input_, 2, dim=1), dim=0)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        output = F.batch_norm(
            input_, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        output = torch.cat(torch.chunk(output, 2, dim=0), dim=1)

        return output


class PartPool(nn.Module):
    def __init__(self, part_num=1, method='max'):
        super(PartPool, self).__init__()
        pools_dict = {'max': nn.AdaptiveMaxPool2d, 'avg': nn.AdaptiveAvgPool2d}
        self.pool = pools_dict[method]((part_num, 1))

    def forward(self, input_):
        return self.pool(input_)


class PartPools(nn.Module):
    def __init__(self, part_nums=(1, 2, 3), methods=('max', 'avg')):
        super(PartPools, self).__init__()
        self.part_nums = part_nums
        self.methods = methods
        self.pools = nn.ModuleList()
        pools_dict = {'max': nn.AdaptiveMaxPool2d, 'avg': nn.AdaptiveAvgPool2d}
        for part_num in part_nums:
            n_pools = nn.ModuleList()
            for m in methods:
                n_pools.append(pools_dict[m]((part_num, 1)))

            self.pools.append(n_pools)

    def forward(self, input):
        results = []
        for sub_pools in self.pools:
            sub_results = []
            for pool in sub_pools:
                sub_results.append(pool(input))

            results.append(torch.cat(sub_results, 1))

        return results


class CatPooledVectors(nn.Module):
    def forward(self, inputs):
        raise NotImplementedError
        sizes_num = len(inputs[0])






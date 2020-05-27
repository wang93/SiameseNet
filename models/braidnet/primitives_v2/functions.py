import torch
from torch.autograd import Function

__all__ = ['Min2', 'Max2']


class Min2(Function):
    @staticmethod
    def forward(ctx, a, b):
        return torch.min(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Max2(Function):
    @staticmethod
    def forward(ctx, a, b):
        return torch.max(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Min3(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torch.min(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_variables
        return grad_output * b, grad_output * a


class ChannelScaling(Function):
    @staticmethod
    def forward(ctx, alpha, y):
        scaling_factor = alpha.exp()
        z = y * scaling_factor
        ctx.save_for_backward(scaling_factor, z)
        return z

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_z = grad_outputs[0]
        scaling_factor, z = ctx.saved_tensors
        grad_y = grad_z * scaling_factor

        grad_alpha = grad_z * z
        # batchsize = grad_alpha.size(0)

        grad_alpha = grad_alpha.sum(0)
        grad_alpha = grad_alpha.sum((1, 2), keepdim=True)
        # grad_alpha = grad_alpha / float(batchsize)

        return grad_alpha, grad_y


if __name__ == '__main__':
    a = torch.randn((2, 3, 8, 8), requires_grad=True)
    b = torch.nn.Parameter(torch.randn(3, 1, 1))
    c = ChannelScaling().apply(b, a)
    print(a)
    print(b)
    print(c)

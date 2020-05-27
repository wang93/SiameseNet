import torch
from torch.autograd import Function

__all__ = ['Min2', 'Max2', 'ChannelScaling2d', 'ChannelScaling1d']


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


class ChannelScaling2d(Function):
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
        grad_alpha = grad_alpha.sum(0)
        grad_alpha = grad_alpha.sum((1, 2), keepdim=True)

        return grad_alpha, grad_y


class ChannelScaling1d(Function):
    @staticmethod
    def forward(ctx, alpha, y):
        scaling_factor = alpha.exp()
        z = y * scaling_factor
        ctx.save_for_backward(scaling_factor, z)
        print('forward: {0}'.format(id(z)))
        return z

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_z = grad_outputs[0]
        scaling_factor, z = ctx.saved_tensors
        print('backward: {0}'.format(id(z)))
        grad_y = grad_z * scaling_factor
        grad_alpha = grad_z * z
        grad_alpha = grad_alpha.sum(0)
        grad_alpha = grad_alpha.sum((1,), keepdim=True)

        return grad_alpha, grad_y


if __name__ == '__main__':
    a = torch.ones((1, 1, 1, 1), requires_grad=True)
    aa = torch.ones((1, 1, 1, 1), requires_grad=True) * 2

    b = torch.nn.Parameter(torch.ones(1, 1, 1))

    c = ChannelScaling2d().apply(b, a)
    c = c.view(-1).mean()

    cc = ChannelScaling2d().apply(b, aa)
    cc = cc.view(-1).mean()

    c.backward()

    print(b.grad)

    cc.backward()

    print(b.grad)

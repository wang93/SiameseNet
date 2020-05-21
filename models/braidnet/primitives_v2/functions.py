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


if __name__ == '__main__':
    a = torch.tensor(1., requires_grad=True)
    b = torch.tensor(2., requires_grad=True)
    c = Min2.apply(a, b) - 1.5 * Max2.apply(a, b)
    c.backward()
    print(a.grad, b.grad)

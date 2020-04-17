from torch.autograd import Function
from torch.nn import functional as F
import torch


class w_batch_norm(Function):
    """
    We can implement our own custom autograd Functions by subclassing
     torch.autograd.Function and implementing the forward and backward passes
     which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, *args):
        """
     In the forward pass we receive a context object and a Tensor containing the
        input; we must return a Tensor containing the output, and we can use the
        context object to cache objects for use in the backward pass.
     """
        # alpha = args[0]
        # y = args[1]
        # scaling_factor = alpha.exp().data
        # z = y.data * scaling_factor.data
        # ctx.save_for_backward(scaling_factor, z)
        # return z
        running_mean = args[1]
        running_var = args[2]
        weight = args[3]
        bias = args[4]



    @staticmethod
    def backward(ctx, *grad_outputs):
        # def backward(ctx, grad_z):
        """
     In the backward pass we receive the context object and a Tensor containing
        the gradient of the loss with respect to the output produced during the
        forward pass. We can retrieve cached data from the context object, and must
        compute and return the gradient of the loss with respect to the input to the
        forward function.
     """
        grad_z = grad_outputs[0]
        scaling_factor, z = ctx.saved_tensors
        grad_y = grad_z.data * scaling_factor.data

        grad_alpha = grad_z.data * z.data
        batchsize = grad_alpha.size(0)
        # print('ddd')
        # print(grad_alpha.size())
        # print(batchsize)

        grad_alpha = grad_alpha.sum(0).data
        #grad_alpha = grad_alpha.mean(0).data
        grad_alpha = grad_alpha.sum((1, 2), keepdim=True).data
        grad_alpha = grad_alpha.data / float(batchsize)

        grad_alpha.fill_(2.5e-4)

        #print(grad_alpha.mean())
        #grad_alpha += 5e-6
        return grad_alpha, grad_y

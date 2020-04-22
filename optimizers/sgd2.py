from torch.optim.sgd import SGD
from torch.optim.optimizer import required

class SGD2(SGD):
    def __init__(self, params, base_lr=required, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if base_lr is not required and base_lr < 0.0:
            raise ValueError("Invalid base learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(base_lr=base_lr, lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)
from torch import nn
import torch
from torch.optim import SGD, Adam, AdamW
from .sampler import SampleRateSampler


class SRL_BCELoss(nn.Module):
    def __init__(self, sampler: SampleRateSampler, optim='sgd', lr=0.1, momentum=0., weight_decay=0.):
        print('using SRL_BCELoss')
        if not isinstance(sampler, SampleRateSampler):
            raise TypeError

        super(SRL_BCELoss, self).__init__()

        self.bce_loss = nn.BCELoss(reduction='none')
        self.alpha = nn.Parameter(torch.tensor(0.).cuda())
        self.pos_rate = self.alpha.sigmoid()
        self.sampler = sampler
        self.sampler.update(self.pos_rate)

        param_groups = [{'params': [self.alpha]}]
        if optim == "sgd":
            default = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
            optimizer = SGD(param_groups, **default)

        elif optim == 'adam':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             amsgrad=False)

        elif optim == 'amsgrad':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = Adam(param_groups, **default,
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             amsgrad=True)

        elif optim == 'adamw':
            default = {'lr': lr, 'weight_decay': weight_decay}
            optimizer = AdamW(param_groups, **default,
                              betas=(0.9, 0.999),
                              eps=1e-8,
                              amsgrad=False)
        else:
            raise NotImplementedError

        self.optimizer = optimizer

    def forward(self, scores, labels: torch.Tensor):
        losses = self.bce_loss(scores.sigmoid(), labels)
        is_pos = labels.type(torch.bool)
        pos_loss = losses[is_pos].mean()
        neg_loss = losses[~is_pos].mean()
        if torch.isnan(pos_loss):
            loss = neg_loss
        elif torch.isnan((neg_loss)):
            loss = pos_loss
        else:
            loss = (pos_loss + neg_loss) / 2.

        # update pos_rate
        grad = (neg_loss - pos_loss).detach()
        if not torch.isnan(grad):
            self.optimizer.zero_grad()
            self.pos_rate.backward(grad)
            self.optimizer.step()
            self.pos_rate = self.alpha.sigmoid()
            self.sampler.update(self.pos_rate)

        return loss

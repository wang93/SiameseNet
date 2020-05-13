# encoding: utf-8
import time

import torch

from utils.meters import AverageMeter
from utils.serialization import save_best_model, save_current_status
from utils.tensor_section_functions import slice_tensor, tensor_size


class _Trainer:
    def __init__(self, opt, train_loader, evaluator, optimzier, lr_strategy, criterion, summary_writer, best_rank1=-1,
                 best_epoch=0,
                 phase_num=1):
        self.opt = opt
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.model = evaluator.model
        self.optimizer = optimzier
        self.lr_strategy = lr_strategy
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.best_rank1 = best_rank1
        self.best_epoch = best_epoch
        self.phase_num = phase_num

    def train_from(self, done_epoch):
        for epoch in range(done_epoch + 1, self.opt.max_epoch + 1):
            self.train(epoch)

    def train(self, epoch):
        """Note: epoch should start with 1"""

        try:
            if epoch == self.opt.freeze_pretrained_untill:
                print('no longer freeze pretrained params (if there were any pretrained params)!')
                self.model.module.unlable_pretrained()
                self.optimizer = self.model.module.get_optimizer(optim=self.opt.optim,
                                                                 lr=self.opt.lr,
                                                                 momentum=self.opt.momentum,
                                                                 weight_decay=self.opt.weight_decay)
        except AttributeError:
            print('the net does not have \'unlable_pretrained\' method')

        start = time.time()
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.lr_strategy(self.optimizer, epoch)
        for i, inputs in enumerate(self.train_loader):
            data_time.update(time.time() - start)
            # model optimizer
            self._parse_data(inputs)
            self._forward()
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            losses.update(self.loss.item())

            # tensorboard
            global_step = (epoch - 1) * len(self.train_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(self.train_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))

            batch_time.update(time.time() - start)
            start = time.time()

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.6f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))

        if self.opt.eval_step > 0 and epoch % self.opt.eval_step == 0 or epoch == self.opt.max_epoch:
            # savefig = join(self.opt.fig_dir, 'origin') if epoch == self.opt.max_epoch else None
            savefig = True if epoch == self.opt.max_epoch else False
            rank1 = self.evaluator.evaluate(re_ranking=self.opt.re_ranking, savefig=savefig, eval_flip=False)

            if rank1 > self.best_rank1:
                save_best_model(self.model, exp_dir=self.opt.exp_dir, epoch=epoch, rank1=rank1)
                self.best_rank1 = rank1
                self.best_epoch = epoch

        save_current_status(self.model, self.optimizer, self.opt.exp_dir, epoch)

        if epoch == self.opt.max_epoch:
            print('Best rank-1 {:.1%}, achieved at epoch {}'.format(self.best_rank1, self.best_epoch))
            self.evaluator.evaluate(re_ranking=self.opt.re_ranking, savefig=True, eval_flip=True)

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError

    def _extract_feature(self, data):
        raise NotImplementedError

    def _compare_feature(self, features):
        raise NotImplementedError


class BraidPairTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super(BraidPairTrainer, self).__init__(*args, **kwargs)

    def _parse_data(self, inputs):
        (imgs_a, pids_a, _), (imgs_b, pids_b, _) = inputs

        target = [1. if a == b else 0. for a, b in zip(pids_a, pids_b)]
        self.data = (imgs_a.cuda(), imgs_b.cuda())
        self.target = torch.tensor(target).cuda().unsqueeze(1)

    def _extract_feature(self, data):
        return self.model(data, None, mode='extract')

    def _compare_feature(self, features):
        return self.model(*features, mode='metric').squeeze()

    def _forward(self):
        if self.phase_num == 1:
            score = self.model(*self.data, mode='normal')

        elif self.phase_num == 2:
            feat_a = self.model(self.data[0], mode='extract')
            feat_b = self.model(self.data[1], mode='extract')
            score = self.model(feat_a, feat_b, mode='metric')

        else:
            raise ValueError

        self.loss = self.criterion(score, self.target)

    def _backward(self):
        self.loss.backward()
        self.model.module.correct_grads()


class BraidCrossTrainer(BraidPairTrainer):
    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _compare_feature(self, features):
        # only compute the lower triangular of the distmat

        n = tensor_size(features, dim=0)
        a_indices, b_indices = torch.tril_indices(n, n)
        scores_l = self.model(slice_tensor(features, a_indices),
                              slice_tensor(features, b_indices),
                              mode='metric').squeeze()

        score_mat = torch.zeros((n, n), device=scores_l.device, dtype=scores_l.dtype)
        score_mat[a_indices, b_indices] = scores_l
        score_mat[b_indices, a_indices] = scores_l

        return score_mat

    def _forward(self):
        if self.phase_num == 1:
            raise NotImplementedError('In most cases, it will waste too much computation.')

        elif self.phase_num == 2:
            features = self._extract_feature(self.data)
            score_mat = self._compare_feature(features)

        else:
            raise ValueError

        self.loss = self.criterion(score_mat, self.target)

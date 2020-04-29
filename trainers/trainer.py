# encoding: utf-8
import time
from os.path import join

import torch

from utils.meters import AverageMeter
from utils.serialization import save_best_model, save_current_status


class cls_tripletTrainer:
    def __init__(self, opt, evaluator, optimzier, lr_strategy, criterion, summary_writer, best_rank1=-1, best_epoch=0):
        self.opt = opt
        self.evaluator = evaluator
        self.model = evaluator.model
        self.optimizer = optimzier
        self.lr_strategy = lr_strategy
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.best_rank1 = best_rank1
        self.best_epoch = best_epoch

    def train(self, epoch, data_loader):
        """Note: epoch should start with 1"""

        try:
            if epoch == self.opt.freeze_pretrained_untill:
                print('no longer freeze pretrained params (if there were any pretrained params)!')
                self.model.module.unlable_pretrained()
                optimizer = self.model.module.get_optimizer(optim=self.opt.optim,
                                                            lr=self.opt.lr,
                                                            momentum=self.opt.momentum,
                                                            weight_decay=self.opt.weight_decay)
                self.optimizer = optimizer
        except AttributeError:
            print('the net does not have \'unlable_pretrained\' method')


        start = time.time()
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.lr_strategy(epoch)
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            self._forward()
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            losses.update(self.loss.item())

            # tensorboard
            global_step = (epoch - 1) * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), global_step)
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))

            batch_time.update(time.time() - start)
            start = time.time()

        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))

        save_current_status(self.model, self.optimizer, self.opt.exp_dir, epoch)

        if self.opt.eval_step > 0 and epoch % self.opt.eval_step == 0 or epoch == self.opt.max_epoch:
            savefig = join(self.opt.savefig, 'origin') if epoch == self.opt.max_epoch else None
            rank1 = self.evaluator.evaluate(re_ranking=self.opt.re_ranking, savefig=savefig)

            is_best = rank1 > self.best_rank1
            if is_best:
                save_best_model(self.model, exp_dir=self.opt.exp_dir, epoch=epoch, rank1=rank1)
                self.best_rank1 = rank1
                self.best_epoch = epoch

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat = self.model(self.data)
        self.loss = self.criterion(score, feat, self.target)

    def _backward(self):
        self.loss.backward()


class braidTrainer(cls_tripletTrainer):
    def __init__(self, *args, **kwargs):
        super(braidTrainer, self).__init__(*args, **kwargs)

    def _parse_data(self, inputs):
        (imgs_a, pids_a, _), (imgs_b, pids_b, _) = inputs

        target = [1. if a == b else 0. for a, b in zip(pids_a, pids_b)]
        self.data = (imgs_a.cuda(), imgs_b.cuda())
        self.target = torch.tensor(target).cuda().unsqueeze(1)

    def _forward(self):
        score = self.model(*self.data)
        self.loss = self.criterion(score, self.target)

    def _backward(self):
        self.loss.backward()
        self.model.module.correct_grads()


class braid_tripletTrainer(braidTrainer):
    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()
        self.n = len(pids)

    def _slice_tensor(self, data, indices):
        if isinstance(data, torch.Tensor):
            return data[indices]
        elif isinstance(data, (list, tuple)):
            return [self._slice_tensor(d, indices) for d in data]
        elif isinstance(data, dict):
            return {k: self._slice_tensor(v, indices) for k, v in data.items()}
        else:
            raise TypeError('type {0} is not supported'.format(type(data)))

    def _extract_feature(self):
        self.features = self.model(self.data, None, mode='extract')
        # self.dims_num = len(self.features.size())

    def _compare_feature(self):
        """has been optimized to save half of the time"""
        # only compute the lower triangular of the distmat
        a_indices, b_indices = torch.tril_indices(self.n, self.n)
        dists_l = - self.model(self._slice_tensor(self.features, a_indices),
                               self._slice_tensor(self.features, b_indices),
                               mode='metric').squeeze()

        distmat = torch.zeros((self.n, self.n), device=dists_l.device, dtype=dists_l.dtype)
        distmat[a_indices, b_indices] = dists_l
        distmat[b_indices, a_indices] = dists_l

        self.distmat = distmat
        # repeat_param = [1, ]*self.dims_num
        # repeat_param[0] = self.n
        # self.distmat = - self.model(self.features.repeat(*repeat_param),
        #                             self.features.repeat_interleave(*repeat_param),
        #                             mode='metric').reshape(self.n, self.n)

    def _forward(self):
        self._extract_feature()
        self._compare_feature()
        self.loss = self.criterion(self.distmat, self.target)

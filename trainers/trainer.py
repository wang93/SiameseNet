# encoding: utf-8
import time

import numpy as np
import torch

from utils.meters import AverageMeter
from utils.serialization import save_best_model, save_current_status, get_best_model
from utils.standard_actions import print_time
from utils.tensor_section_functions import slice_tensor, tensor_size, tensor_cuda, tensor_cpu


class _Trainer:
    def __init__(self, opt, train_loader, evaluator, optimzier, lr_strategy,
                 criterion, summary_writer, phase_num=1, done_epoch=0):
        self.opt = opt
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.model = evaluator.model
        self.optimizer = optimzier
        self.lr_strategy = lr_strategy
        self.criterion = criterion
        self.summary_writer = summary_writer
        _, best_epoch, best_rank1 = get_best_model(opt.exp_dir)
        self.best_rank1 = best_rank1
        self.best_epoch = best_epoch
        self.phase_num = phase_num
        self.done_epoch = done_epoch

    @print_time
    def continue_train(self):
        while self.done_epoch < self.opt.max_epoch:
            self._train(self.done_epoch + 1)
            self.done_epoch += 1

        if self.opt.savefig:
            self.visualize_best()

    def _adapt_to_best(self):
        best_state_dict, best_epoch, best_rank1 = get_best_model(self.opt.exp_dir)
        self.model.module.load_state_dict(best_state_dict)
        return best_epoch, best_rank1

    @print_time
    def visualize_best(self):
        best_epoch, best_rank1 = self._adapt_to_best()
        print('visualization based on the best model (rank-1 {:.1%}, achieved at epoch {}).'
              .format(best_rank1, best_epoch))
        self.evaluator.visualize(re_ranking=self.opt.re_ranking, eval_flip=False)
        self.evaluator.visualize(re_ranking=self.opt.re_ranking, eval_flip=True)
        print('The whole process should be terminated.')

    def evaluate_best(self, eval_flip=None):
        best_epoch, best_rank1 = self._adapt_to_best()
        print('evaluation based on the best model (rank-1 {:.1%}, achieved at epoch {}).'
              .format(best_rank1, best_epoch))
        self.evaluate(eval_flip)
        print('The whole process should be terminated.')

    def _train(self, epoch):
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
            rank1 = self.evaluate(eval_flip=False)

            if rank1 > self.best_rank1:
                save_best_model(self.model, exp_dir=self.opt.exp_dir, epoch=epoch, rank1=rank1)
                self.best_rank1 = rank1
                self.best_epoch = epoch

        save_current_status(self.model, self.optimizer, self.opt.exp_dir, epoch, self.opt.eval_step)

    @print_time
    def evaluate(self, eval_flip=None):
        if eval_flip is None:
            self.evaluator.evaluate(re_ranking=self.opt.re_ranking, eval_flip=False)
            self.evaluator.evaluate(re_ranking=self.opt.re_ranking, eval_flip=True)
        else:
            rank1 = self.evaluator.evaluate(re_ranking=self.opt.re_ranking, eval_flip=eval_flip)
            return rank1

    def _get_feature_with_id(self, dataloader):
        with torch.no_grad():
            fun = lambda d: self.model(d, None, mode='extract')
            records = [(tensor_cpu(fun(tensor_cuda(data))), identity) for data, identity, _ in dataloader]

            features = []
            ids = []
            for features_, ids_ in records:
                features.extend(features_.tolist())
                ids.extend(ids_)

        return np.array(features), np.array(ids)

    @print_time
    def check_discriminant(self):
        if self.opt.dataset is not 'market1501':
            raise NotImplementedError
        from dataset.attributes import get_market_attributes
        from sklearn import svm
        from sklearn.metrics import confusion_matrix
        from pprint import pprint

        features, ids = self._get_feature_with_id(self.train_loader)

        sample_num = len(features)
        split_border = sample_num // 2 + 1
        features_train = features[:split_border]
        features_test = features[split_border:]

        ids = list(ids)
        attributes = get_market_attributes(set_name='train')

        attribute_ids = attributes.pop('image_index')
        index_map = [attribute_ids.index(i) for i in ids]

        attributes_new = dict()
        for key, label in attributes.items():
            label_new = [label[i] for i in index_map]
            attributes_new[key] = np.array(label_new)

        for key, labels in attributes_new.items():
            labels_train = labels[:split_border]
            labels_test = labels[split_border:]
            classes = set(labels)
            for class_ in classes:
                print('checking the discriminant for the label {0} of {1}'.format(class_, key))
                hitted_train = (labels_train == class_).astype(int)
                hitted_test = (labels_test == class_).astype(int)

                model = svm.SVC(kernel='linear')
                model.fit(features_train, hitted_train)
                prediction = model.predict(features_test)

                cm = confusion_matrix(y_pred=prediction, y_true=hitted_test)
                print('confusion matrix:')
                pprint(cm)
                print()

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
    def _parse_data(self, inputs):
        (imgs_a, pids_a, _), (imgs_b, pids_b, _) = inputs

        target = [1. if a == b else 0. for a, b in zip(pids_a, pids_b)]
        self.data = (imgs_a.cuda(), imgs_b.cuda())
        self.target = torch.tensor(target).cuda().unsqueeze(1)

    def _extract_feature(self, data):
        return self.model(data, mode='extract')

    def _compare_feature(self, *features):
        return self.model(*features, mode='metric')

    def _forward(self):
        if self.phase_num == 1:
            score = self.model(*self.data, mode='normal')

        elif self.phase_num == 2:
            feat_a = self._extract_feature(self.data[0])
            feat_b = self._extract_feature(self.data[1])
            score = self._compare_feature(feat_a, feat_b)

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

        if len(scores_l.size()) == 1:
            score_mat = torch.zeros((n, n), device=scores_l.device, dtype=scores_l.dtype)
        elif len(scores_l.size()) == 2:
            score_mat = torch.zeros((n, n, scores_l.size()[1]), device=scores_l.device, dtype=scores_l.dtype)
        else:
            raise NotImplementedError

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


class NormalTrainer(_Trainer):
    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        if self.phase_num == 1:
            predictions = self._extract_feature(self.data)

        elif self.phase_num == 2:
            raise NotImplementedError

        else:
            raise ValueError

        self.loss = self.criterion(predictions, self.target)

    def _backward(self):
        self.loss.backward()

    def _extract_feature(self, data):
        return self.model(self.data, mode='extract')

    def _compare_feature(self, features):
        raise NotImplementedError

# encoding: utf-8
import os

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from utils.re_ranking import re_ranking as re_ranking_func

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from collections import defaultdict
from random import choice as randchoice
from time import time as curtime

from utils.tensor_section_functions import *

from utils.adaptive_batchsize import get_optimized_batchsize
from datasets.samplers import PosNegPairSampler


class ReIDEvaluator:
    def __init__(self, model, queryloader, galleryloader, queryFliploader, galleryFliploader, phase_num=1, minors_num=0,
                 ranks=(1, 2, 4, 5, 8, 10, 16, 20)):
        self.model = model
        self.queryloader = queryloader
        self.galleryloader = galleryloader
        self.queryFliploader = queryFliploader
        self.galleryFliploader = galleryFliploader
        self.ranks = ranks
        self.phase_num = phase_num
        self.minors_num = minors_num

        # self.batch_size = self.queryloader.batch_size

    def save_incorrect_pairs(self, distmat, g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            # for j in range(10):
            #     index = indices[i][j]
            #     if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
            #         continue
            #     else:
            #         break

            # if g_pids[index] == q_pids[i]:
            #     continue
            fig, axes = plt.subplots(1, 11, figsize=(12, 8))
            img = self.queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            cur_loc = 1
            for gallery_index in indices[i]:
                if g_camids[gallery_index] == q_camids[i] and g_pids[gallery_index] == q_pids[i]:
                    continue
                cur_loc += 1
                # gallery_index = indices[i][j]
                img = self.galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[cur_loc].set_title(g_pids[gallery_index])
                axes[cur_loc].set_axis_off()
                axes[cur_loc].imshow(img)
                if cur_loc == 11:
                    break

            fig.savefig(os.path.join(savefig, '%d.png' % q_pids[i]))
            plt.close(fig)

    def measure_scores(self, distmat, q_pids, g_pids, q_camids, g_camids, immidiate=True):
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)
        threshold, eer = self.eer_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        if immidiate:
            print("----------- Evaluation Report ----------")
            print("mAP: {:.1%}".format(mAP))
            print("CMC curve")
            for r in self.ranks:
                print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            print("EER: {:.1%}, corresponding threshold: {:.3f}".format(eer, threshold))
            print("----------------------------------------")
            return cmc[0]
        else:
            return mAP, cmc, eer, threshold

    def measure_scores_on_minors(self, distmat_all, q_pids_all, g_pids_all, q_camids_all, g_camids_all):
        print('average evaluation results on {0} testset minors'.format(self.minors_num))
        qpid2index = defaultdict(list)
        gpid2index = defaultdict(list)
        q_pids_all = q_pids_all.tolist()
        g_pids_all = g_pids_all.tolist()
        for i, qpid in enumerate(q_pids_all):
            qpid2index[qpid].append(i)
        for i, gpid in enumerate(g_pids_all):
            gpid2index[gpid].append(i)

        pids = list(set(q_pids_all))

        q_pids = torch.Tensor(pids)
        g_pids = torch.Tensor(pids)

        cmcs, mAPs, thresholds, eers = [], [], [], []
        for _ in range(self.minors_num):
            q_indices = torch.LongTensor([randchoice(qpid2index[pid]) for pid in pids])
            g_indices = torch.LongTensor([randchoice(gpid2index[pid]) for pid in pids])
            q_camids = q_camids_all[q_indices]
            g_camids = g_camids_all[g_indices]
            distmat = distmat_all[q_indices, :][:, g_indices]

            mAP_, cmc_, eer_, threshold_ = self.measure_scores(distmat, q_pids, g_pids, q_camids, g_camids, immidiate=False)

            mAPs.append(mAP_)
            cmcs.append(cmc_)
            thresholds.append(threshold_)
            eers.append(eer_)

        mAP = np.mean(mAPs)
        cmc = np.mean(cmcs, 0)
        threshold = np.mean(thresholds)
        eer = np.mean(eers)

        print("---------- Evaluation Report ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in self.ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("EER: {:.1%}, with threshold: {:.3f}".format(eer, threshold))
        print("----------------------------------------")

        return cmc[0]

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1]) 
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices] == q_camids.view([num_q, -1])))
        #keep = g_camids[indices]  != q_camids.view([num_q, -1])
        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank+1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eer_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids):
        num_q, num_g = distmat.size()
        scores, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1])
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices] == q_camids.view([num_q, -1])))
        #keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        scores_ = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            s = scores[i][keep[i]]
            if m.any():
                results.append(m)
                scores_.append(s)
        matches = torch.cat(results, dim=0).float()
        scores_ = - torch.cat(scores_, dim=0)

        fpr, tpr, thresholds = roc_curve(matches, scores_, pos_label=1.)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)

        return thresh, eer

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _compare_features(self, a, b):
        l_a = tensor_size(a, 0)
        l_b = tensor_size(b, 0)
        score_mat = torch.zeros(l_a, l_b)

        tasks = [np.arange(l_a).repeat(l_b), np.tile(np.arange(l_b), l_a)]

        # cur_idx_a = -1
        with torch.no_grad():
            fun = lambda a, b: self.model(a, b, mode='metric').view(-1)
            batch_size = get_optimized_batchsize(fun, slice_tensor(a, [0]), slice_tensor(b, [0]))
            # batch_size = min(batch_size, l_b)

            task_num = l_a * l_b
            for start in range(0, task_num, batch_size):
                end = min(start + batch_size, task_num)
                a_indices = tasks[0][start:end]
                b_indices = tasks[1][start:end]
                sub_fa = slice_tensor(a, a_indices)
                sub_fb = slice_tensor(b, b_indices)
                sub_fa, sub_fb = tensor_cuda((sub_fa, sub_fb))
                scores = fun(sub_fa, sub_fb).cpu()
                score_mat[a_indices, b_indices] = scores

        return score_mat

    def _compare_images(self, loader_a, loader_b):
        l_a = len(loader_a)
        l_b = len(loader_b)
        score_mat = torch.zeros(l_a, l_b)

        tasks = [np.arange(l_a).repeat(l_b), np.tile(np.arange(l_b), l_a)]

        # cur_idx_a = -1
        with torch.no_grad():
            fun = lambda a, b: self.model(a, b, mode='normal').view(-1)
            one_ima = slice_tensor(next(iter(loader_a))[0], [0])
            one_imb = slice_tensor(next(iter(loader_b))[0], [0])
            batch_size = get_optimized_batchsize(fun, one_ima, one_imb)
            del one_ima, one_imb
            self._change_batchsize(loader_a, batch_size)
            self._change_batchsize(loader_b, batch_size)
            self._change_sampler(loader_a, tasks[0])
            self._change_sampler(loader_b, tasks[1])
            # batch_size = min(batch_size, l_b)

            task_num = l_a * l_b
            cur_idx = 0
            for (ima_s, _, _), (imb_s, _, _) in zip(loader_a, loader_b):
                ima_s = tensor_cuda(ima_s)
                imb_s = tensor_cuda(imb_s)
                scores = fun(ima_s, imb_s).cpu()
                end = min(cur_idx + batch_size, task_num)
                score_mat[tasks[0][cur_idx:end], tasks[1][cur_idx:end]] = scores
                cur_idx = end

        return score_mat

        # with torch.no_grad():
        #     fun = lambda a, b: self.model(a, b, mode='normal').view(-1)
        #     one_ima = slice_tensor(next(iter(loader_a))[0], [0])
        #     one_imb = slice_tensor(next(iter(loader_b))[0], [0])
        #     batch_size = get_max_batchsize(fun, one_ima, one_imb)
        #     batch_size = min(batch_size, l_b)
        #     self._change_batchsize(loader_a, 1)
        #     self._change_batchsize(loader_b, batch_size)
        #
        #     batches_b = [batch for batch, _, _ in loader_b]
        #
        #     for cur_idx_a, (ima, _, _) in enumerate(loader_a):
        #         cur_idx_b = 0
        #         ima_s = tensor_repeat(ima, dim=0, num=batch_size, interleave=True)
        #         ima_s = tensor_cuda(ima_s)
        #         n_a = batch_size
        #         for imb_s in batches_b:
        #             imb_s = tensor_cuda(imb_s)
        #             n_b = tensor_size(imb_s, 0)
        #             if n_a != n_b:
        #                 ima_s = tensor_repeat(ima, dim=0, num=n_b, interleave=True)
        #                 ima_s = tensor_cuda(ima_s)
        #                 n_a = n_b
        #
        #             scores = fun(ima_s, imb_s).cpu()
        #             score_mat[cur_idx_a, cur_idx_b:cur_idx_b + n_b] = scores
        #
        #             cur_idx_b += n_b
        #
        # return score_mat

    @staticmethod
    def _change_batchsize(dataloader, batch_size):
        if isinstance(dataloader.sampler, PosNegPairSampler):
            raise TypeError('can not change the batchsize of dataloader with pos_neg_pair_sampler')
        dataloader._DataLoader__initialized = False
        dataloader.batch_size = batch_size
        dataloader.batch_sampler.batch_size = batch_size
        dataloader._DataLoader__initialized = True

    @staticmethod
    def _change_sampler(dataloader, sampler: list):
        if isinstance(dataloader.sampler, PosNegPairSampler):
            raise TypeError('can not change the sampler of dataloader with pos_neg_pair_sampler')
        dataloader._DataLoader__initialized = False
        dataloader.sampler = sampler
        dataloader.batch_sampler.sampler = sampler
        dataloader.batch_sampler.drop_last = False
        dataloader._DataLoader__initialized = True

    def _get_feature(self, dataloader):
        with torch.no_grad():
            fun = lambda d: self.model(d, None, mode='extract')
            batch_size = get_optimized_batchsize(fun, slice_tensor(next(iter(dataloader))[0], [0]))
            batch_size = min(batch_size, len(dataloader))
            self._change_batchsize(dataloader, batch_size)

            features = [tensor_cpu(fun(tensor_cuda(data))) for data, _, _ in dataloader]
            features = cat_tensors(features, dim=0)  # torch.cat(features, dim=0)
        return features

    def evaluate(self, eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        if eval_flip:
            print('**** evaluate with flip images ****')

        q_pids, q_camids = [], []
        g_pids, g_camids = [], []
        for queries in self.queryloader:
            _, pids, camids = self._parse_data(queries)
            q_pids.extend(pids)
            q_camids.extend(camids)

        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        for galleries in self.galleryloader:
            _, pids, camids = self._parse_data(galleries)
            g_pids.extend(pids)
            g_camids.extend(camids)

        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        start = curtime()

        with torch.no_grad():

            if self.phase_num == 1:
                q_g_similarity = self._compare_images(self.queryloader, self.galleryloader)

                if eval_flip:
                    q_g_similarity += self._compare_images(self.queryloader, self.galleryFliploader)
                    q_g_similarity += self._compare_images(self.queryFliploader, self.galleryloader)
                    q_g_similarity += self._compare_images(self.queryFliploader, self.galleryFliploader)
                    q_g_similarity /= 4.0

            elif self.phase_num == 2:
                '''phase one'''
                query_features = self._get_feature(self.queryloader)
                gallery_features = self._get_feature(self.galleryloader)

                '''phase two'''
                q_g_similarity = self._compare_features(query_features, gallery_features)

                if not eval_flip:
                    del gallery_features, query_features
                else:
                    query_flip_features = self._get_feature(self.queryFliploader)
                    q_g_similarity += self._compare_features(query_flip_features, gallery_features)
                    del gallery_features
                    gallery_flip_features = self._get_feature(self.galleryFliploader)
                    q_g_similarity += self._compare_features(query_flip_features, gallery_flip_features)
                    del query_flip_features
                    q_g_similarity += self._compare_features(query_features, gallery_flip_features)
                    del gallery_flip_features, query_features
                    q_g_similarity /= 4.0

            else:
                raise ValueError

        end = curtime()
        print('it costs {:.3f} s to compute similarity matrix'
              .format(end - start))

        if re_ranking:
            raise NotImplementedError('Not recommended, as it costs too much time.')
        else:
            distmat = -q_g_similarity

        if self.minors_num <= 0:
            rank1 = self.measure_scores(distmat, q_pids, g_pids, q_camids, g_camids, immidiate=True)
        else:
            rank1 = self.measure_scores_on_minors(distmat, q_pids, g_pids, q_camids, g_camids)

        if savefig:
            print("Saving visualization fingures")
            self.save_incorrect_pairs(distmat.numpy(), g_pids.numpy(), q_pids.numpy(), g_camids.numpy(),
                                      q_camids.numpy(), savefig)

        return rank1

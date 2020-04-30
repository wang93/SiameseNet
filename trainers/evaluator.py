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

from utils.tensor_section_functions import tensor_cuda, tensor_repeat, tensor_size, cat_tensors, \
    split_tensor, tensor_attr, slice_tensor

from utils.adaptive_batchsize import get_max_batchsize


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
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes = plt.subplots(1, 11, figsize=(12, 8))
            img = self.queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = self.galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j+1].set_title(g_pids[gallery_index])
                axes[j+1].set_axis_off()
                axes[j+1].imshow(img)
            fig.savefig(os.path.join(savefig, '%d.png' %q_pids[i]))
            plt.close(fig)

    def measure_scores(self, distmat, q_pids, g_pids, q_camids, g_camids, immidiate=True):
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)
        threshold, eer = self.eer_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        if immidiate:
            print("---------- Performance Report ----------")
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
        print('****measure performance by averaging the performance scores on {0} testset minors****'.format(self.minors_num))
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

        print("---------- Performance Report ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in self.ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("EER: {:.1%}, corresponding threshold: {:.3f}".format(eer, threshold))
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

    # def _compare_images(self, *inputs):
    #     with torch.no_grad():
    #         sample_num = tensor_size(inputs, dim=0)
    #         fun = lambda i: self.model(*i, mode='normal')
    #         batch_size = get_max_batchsize(fun, slice_tensor(inputs, [0]))
    #         batch_size = min(batch_size, sample_num)
    #
    #         scores = [fun(tensor_cuda(sub)) for sub in split_tensor(inputs, dim=0, split_size=batch_size)]
    #         scores = torch.cat(scores, dim=0)
    #         #scores = self.model(*inputs, mode='normal')
    #     return scores.cpu()

    def _compare(self, a, b, mode):
        l_a = tensor_size(a, 0)
        l_b = tensor_size(b, 0)
        score_mat = torch.zeros(l_a, l_b, dtype=tensor_attr(a, 'dtype'), device=tensor_attr(a, 'device'))
        cur_idx_a = -1

        with torch.no_grad():
            fun = lambda a, b: self.model(a, b, mode=mode).view(-1)
            batch_size = get_max_batchsize(fun, slice_tensor(a, [0]), slice_tensor(b, [0]))
            batch_size = min(batch_size, l_b)
            # print('when comparing features in evaluator, the maximum batchsize is {0}'.format(batch_size))
            for sub_fa in split_tensor(a, dim=0, split_size=1):
                cur_idx_a += 1
                cur_idx_b = 0
                sub_fa_s = tensor_repeat(sub_fa, dim=0, num=batch_size, interleave=True)
                sub_fa_s = tensor_cuda(sub_fa_s)
                n_a = batch_size
                for sub_fb in split_tensor(b, dim=0, split_size=batch_size):
                    sub_fb = tensor_cuda(sub_fb)
                    n_b = tensor_size(sub_fb, 0)
                    if n_a != n_b:
                        sub_fa_s = tensor_repeat(sub_fa, dim=0, num=n_b, interleave=True)
                        sub_fa_s = tensor_cuda(sub_fa_s)

                    # scores = self.model(sub_fa_s, sub_fb, mode='metric').view(-1).cpu()
                    scores = fun(sub_fa_s, sub_fb).cpu()
                    score_mat[cur_idx_a, cur_idx_b:cur_idx_b + n_b] = scores

                    cur_idx_b += n_b

        return score_mat

    def _get_feature(self, dataloader):
        with torch.no_grad():
            fun = lambda d: self.model(d, None, mode='extract')
            batch_size = get_max_batchsize(fun, dataloader.dataset[0][0])
            batch_size = min(batch_size, len(dataloader))
            dataloader.batch_size = batch_size
            dataloader.batch_sampler.batch_size = batch_size

            features = [fun(tensor_cuda(data)).cpu() for data, _, _ in dataloader]
            features = cat_tensors(features, dim=0)  # torch.cat(features, dim=0)
        return features

    # def _extract_feature(self, ims):
    #     with torch.no_grad():
    #         ims = ims.cuda()
    #         features = self.model(ims, None, mode='extract')
    #     return tensor_cpu(features)  # features.cpu()

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
                galleries_all = [galleries for galleries, _, _ in self.galleryloader]
                queries_all = [queries for queries, _, _ in self.queryloader]

                q_g_similarity = self._compare(queries_all, galleries_all, mode='normal')

                if not eval_flip:
                    del galleries_all, queries_all
                else:
                    flip_galleries_all = [galleries for galleries, _, _ in self.galleryFliploader]
                    q_g_similarity += self._compare(queries_all, flip_galleries_all, mode='normal')
                    del queries_all
                    flip_queries_all = [queries for queries, _, _ in self.queryFliploader]
                    q_g_similarity += self._compare(flip_queries_all, galleries_all, mode='normal')
                    del galleries_all
                    q_g_similarity += self._compare(flip_queries_all, flip_galleries_all, mode='normal')
                    del flip_queries_all, flip_galleries_all
                    q_g_similarity /= 4.0

            elif self.phase_num == 2:
                '''phase one'''
                query_features = self._get_feature(self.queryloader)
                gallery_features = self._get_feature(self.galleryloader)

                '''phase two'''
                q_g_similarity = self._compare(query_features, gallery_features, 'metric')

                if not eval_flip:
                    del gallery_features, query_features
                else:
                    query_flip_features = self._get_feature(self.queryFliploader)
                    q_g_similarity += self._compare(query_flip_features, gallery_features, 'metric')
                    del gallery_features
                    gallery_flip_features = self._get_feature(self.galleryFliploader)
                    q_g_similarity += self._compare(query_flip_features, gallery_flip_features, 'metric')
                    del query_flip_features
                    q_g_similarity += self._compare(query_features, gallery_flip_features, 'metric')
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

        if savefig:
            print("Saving visualization fingures")
            self.save_incorrect_pairs(distmat.numpy(), g_pids.numpy(), q_pids.numpy(), g_camids.numpy(),
                                      q_camids.numpy(), savefig)

        if self.minors_num <= 0:
            rank1 = self.measure_scores(distmat, q_pids, g_pids, q_camids, g_camids, immidiate=True)
        else:
            rank1 = self.measure_scores_on_minors(distmat, q_pids, g_pids, q_camids, g_camids)
        return rank1

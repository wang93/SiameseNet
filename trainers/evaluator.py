# encoding: utf-8
import os

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from trainers.re_ranking import re_ranking as re_ranking_func

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from collections import defaultdict
from random import choice as randchoice
from time import time as curtime


class ResNetEvaluator:
    def __init__(self, model, queryloader, galleryloader, queryFliploader, galleryFliploader, minors_num=0, ranks=(1, 2, 4, 5, 8, 10, 16, 20)):
        self.model = model
        self.queryloader = queryloader
        self.galleryloader = galleryloader
        self.queryFliploader = queryFliploader
        self.galleryFliploader = galleryFliploader
        self.ranks = ranks
        self.minors_num = minors_num

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

    def evaluate(self, eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        if eval_flip:
            print('****evaluate based on flip-fused features****')

        qf, q_pids, q_camids = [], [], []
        for inputs0, inputs1 in zip(self.queryloader, self.queryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for inputs0, inputs1 in zip(self.galleryloader, self.galleryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
                
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            print('****evaluate with re-ranked distance matrix****')
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
        else:
            distmat = q_g_dist 

        if savefig:
            print("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        if self.minors_num <= 0:
            rank1 = self.measure_scores(distmat, q_pids, g_pids, q_camids, g_camids, immidiate=True)
        else:
            rank1 = self.measure_scores_on_minors(distmat, q_pids, g_pids, q_camids, g_camids)

        return rank1

    def measure_scores(self, distmat, q_pids, g_pids, q_camids, g_camids, immidiate=True):
        #print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        #print("Computing EER and corresponding threshold")
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

        #mAPs = torch.cat(mAPs, dim=0)
        #cmcs = torch.cat(cmcs, dim=0)
        #thresholds = torch.cat(thresholds, dim=0)
        #eers = torch.cat(eers, dim=0)

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

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

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


class BraidEvaluator(ResNetEvaluator):
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
        num_q, num_g = len(q_pids), len(g_pids)
        q_g_similarity = torch.zeros((num_q, num_g))
        with torch.no_grad():
            galleries_all = [galleries for galleries, _, _ in self.galleryloader]

            cur_query_index = -1
            for queries in self.queryloader:
                q_features, _, _ = self._parse_data(queries)
                for q_feature in q_features:
                    cur_query_index += 1
                    cur_gallery_index = 0
                    for galleries in galleries_all:
                        g_features = galleries.cuda()
                        e = cur_gallery_index + g_features.size(0)
                        features_of_a_query = q_feature.expand_as(g_features)
                        scores = self._forward(features_of_a_query, g_features).view(-1).cpu()
                        q_g_similarity[cur_query_index, cur_gallery_index:e] = scores
                        cur_gallery_index = e

            if not eval_flip:
                del galleries_all
            else:
                cur_query_index = -1
                for queries in self.queryFliploader:
                    q_features, _, _ = self._parse_data(queries)
                    for q_feature in q_features:
                        cur_query_index += 1
                        cur_gallery_index = 0
                        for galleries in galleries_all:
                            g_features = galleries.cuda()
                            e = cur_gallery_index + g_features.size(0)
                            features_of_a_query = q_feature.expand_as(g_features)
                            scores = self._forward(features_of_a_query, g_features).view(-1).cpu()
                            q_g_similarity[cur_query_index, cur_gallery_index:e] += scores
                            cur_gallery_index = e

                del galleries_all
                flip_galleries_all = [galleries for galleries, _, _ in self.galleryFliploader]

                cur_query_index = -1
                for queries in self.queryFliploader:
                    q_features, _, _ = self._parse_data(queries)
                    for q_feature in q_features:
                        cur_query_index += 1
                        cur_gallery_index = 0
                        for galleries in flip_galleries_all:
                            g_features = galleries.cuda()
                            e = cur_gallery_index + g_features.size(0)
                            features_of_a_query = q_feature.expand_as(g_features)
                            scores = self._forward(features_of_a_query, g_features).view(-1).cpu()
                            q_g_similarity[cur_query_index, cur_gallery_index:e] += scores
                            cur_gallery_index = e

                cur_query_index = -1
                for queries in self.queryloader:
                    q_features, _, _ = self._parse_data(queries)
                    for q_feature in q_features:
                        cur_query_index += 1
                        cur_gallery_index = 0
                        for galleries in flip_galleries_all:
                            g_features = galleries.cuda()
                            e = cur_gallery_index + g_features.size(0)
                            features_of_a_query = q_feature.expand_as(g_features)
                            scores = self._forward(features_of_a_query, g_features).view(-1).cpu()
                            q_g_similarity[cur_query_index, cur_gallery_index:e] += scores
                            cur_gallery_index = e

                del flip_galleries_all
                q_g_similarity /= 4.0

        end = curtime()
        print('it costs {:.3f} s to compute similarity matrix'
              .format(end-start))

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

    def _forward(self, *inputs):
        with torch.no_grad():
            scores = self.model(*inputs)
        return scores.cpu()


class BraidEvaluator_2Phases(ResNetEvaluator):
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

        query_features = self._get_feature(self.queryloader)
        gallery_features = self._get_feature(self.galleryloader)

        q_g_similarity = self._compare_feature(query_features, gallery_features)

        if not eval_flip:
            del gallery_features, query_features

        else:
            query_flip_features = self._get_feature(self.queryFliploader)

            q_g_similarity += self._compare_feature(query_flip_features, gallery_features)

            del gallery_features

            gallery_flip_features = self._get_feature(self.galleryFliploader)

            q_g_similarity += self._compare_feature(query_flip_features, gallery_flip_features)

            del query_flip_features

            q_g_similarity += self._compare_feature(query_features, gallery_flip_features)

            del gallery_flip_features

            q_g_similarity /= 4.0

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

    def _forward(self, *inputs):
        raise NotImplementedError('It is a 2-phases evaluator!')

    def _get_feature(self, dataloader):
        features = [self._extract_feature(data) for data, _, _ in dataloader]
        features = torch.cat(features, dim=0)
        return features

    def _extract_feature(self, ims):
        with torch.no_grad():
            ims = ims.cuda()
            features = self.model(ims, None, mode='extract')
        return features.cpu()

    def _compare_feature(self, f_a, f_b):
        dims_num = len(f_a.size())
        n_a = f_a.size()[0]
        n_b = f_b.size()[0]
        repeat_param_a = [1, ] * dims_num
        repeat_param_a[0] = n_b
        repeat_param_b = [1, ] * dims_num
        repeat_param_b[0] = n_a
        with torch.no_grad():
            f_a = f_a.cuda()
            f_b = f_b.cuda()
            score_mat = self.model(f_b.repeat(*repeat_param_b),
                                   f_a.repeat_interleave(*repeat_param_a),
                                   mode='metric').reshape(n_a, n_b)

        return score_mat.cpu()

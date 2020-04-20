from __future__ import absolute_import

from collections import defaultdict

from numpy.random import choice as randchoice
from numpy.random import uniform as randuniform
import torch
import random
from torch.utils.data.sampler import Sampler


class PosNegPairSampler(Sampler):
    def __init__(self, data_source, pos_rate=0.5, iter_num_per_epoch=500):
        super(PosNegPairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.pos_rate = pos_rate
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = iter_num_per_epoch
        #self.num_identities = len(self.pids)

    def __iter__(self):
        print('the length of the sampler is {0}'.format(self.length))
        self.cur_idx = -1
        return self

    def __next__(self):
        self.cur_idx += 1
        print(self.cur_idx)
        if self.cur_idx >= self.length:
            return StopIteration

        if randuniform() < self.pos_rate:
            '''positive pair'''
            pid = randchoice(self.pids)
            candidates = self.index_dic[pid]
            chosen = tuple(randchoice(candidates, size=2, replace=True))
            #label = 1.
        else:
            '''negative pair'''
            pid_pair = tuple(randchoice(self.pids, size=2, replace=False))
            chosen = tuple([randchoice(self.index_dic[pid]) for pid in pid_pair])
            #label = 0.

        return chosen #, label

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.length# len(self.data_source) ** 2


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = randchoice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances

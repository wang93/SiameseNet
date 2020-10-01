from __future__ import absolute_import

from collections import defaultdict

from numpy.random import choice as randchoice
from numpy.random import uniform as randuniform
from torch.utils.data.sampler import Sampler


class SampleRateSampler(Sampler):
    def __init__(self, data_source, sample_num_per_epoch=500*256):
        super(SampleRateSampler, self).__init__(data_source)
        self.data_source = data_source
        # self.alpha = torch.nn.Parameter(torch.tensor(0.))
        self.pos_rate = 0.5
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = sample_num_per_epoch

    def update(self, pos_rate):
        self.pos_rate = pos_rate.cpu().item()

    def __iter__(self):
        self.cur_idx = -1
        return self

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx >= self.length:
            raise StopIteration

        if randuniform() < self.pos_rate:
            '''positive pair'''
            pid = randchoice(self.pids)
            candidates = self.index_dic[pid]
            chosen = tuple(randchoice(candidates, size=2, replace=False))

        else:
            '''negative pair'''
            pid_pair = tuple(randchoice(self.pids, size=2, replace=True))
            chosen = tuple([randchoice(self.index_dic[pid]) for pid in pid_pair])

        return chosen

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.length


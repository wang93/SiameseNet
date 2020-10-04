from __future__ import absolute_import

from collections import defaultdict

from numpy.random import choice as randchoice
from numpy.random import uniform as randuniform
from numpy.random import binomial
from numpy import clip
from torch.utils.data.sampler import Sampler

from queue import Queue
from random import sample as randsample


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
        self.sample_num_per_epoch = sample_num_per_epoch

    def update(self, pos_rate):
        self.pos_rate = pos_rate.cpu().item()

    def __iter__(self):
        self.cur_idx = -1
        return self

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx >= self.sample_num_per_epoch:
            raise StopIteration

        if randuniform() < self.pos_rate:
            '''positive pair'''
            pid = randchoice(self.pids)
            candidates = self.index_dic[pid]
            chosen = tuple(randchoice(candidates, size=2, replace=True))

        else:
            '''negative pair'''
            pid_pair = tuple(randchoice(self.pids, size=2, replace=False))
            chosen = tuple([randchoice(self.index_dic[pid]) for pid in pid_pair])

        return chosen

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return self.sample_num_per_epoch


# class _HalfQueue(object):
#     def __init__(self, elements: list, num=1):
#         num_elements = len(elements)
#         max_recent_num = num_elements // 2
#         self.recent = Queue(maxsize=max_recent_num)
#         self.selection_pool = set(elements)
#         self.num = num
#
#     def _update(self, new_element):
#         self.selection_pool.remove(new_element)
#
#         if self.recent.full():
#             old_element = self.recent.get()
#             self.selection_pool.add(old_element)
#
#         self.recent.put(new_element)
#
#     def select(self):
#         res = randsample(self.selection_pool, self.num)
#         for e in res:
#             self._update(e)
#
#         return res


class SampleRateBatchSampler(SampleRateSampler):
    def __init__(self, data_source, sample_num_per_epoch=500*256, batch_size=1):
        super(SampleRateBatchSampler, self).__init__(data_source, sample_num_per_epoch)

        self.batch_size = batch_size

        # self.pos_agent = _HalfQueue(self.pids, 1)
        # self.neg_agent = _HalfQueue(self.pids, 2)

        self.length = (self.sample_num_per_epoch + self.batch_size - 1) // self.batch_size

    def _get_pos_samples(self, num):
        #pid = self.pos_agent.select()[0]
        pos_pids = randchoice(self.pids, size=num, replace=False)
        pos_samples = [tuple(randchoice(self.index_dic[pid], size=2, replace=True)) for pid in pos_pids]
        return pos_samples

    def _get_neg_samples(self, num):
        #pid_pair = self.neg_agent.select()
        neg_samples = []
        for _ in range(num):
            pid_pair = randchoice(self.pids, size=2, replace=False)
            chosen = tuple([randchoice(self.index_dic[pid]) for pid in pid_pair])
            neg_samples.append(chosen)
        return neg_samples

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx >= self.length:
            raise StopIteration

        # pos_num = binomial(self.batch_size, self.pos_rate)
        pos_num = round(self.batch_size * self.pos_rate)
        pos_num = int(clip(pos_num, 1, self.batch_size-1))
        neg_num = self.batch_size - pos_num

        batch = self._get_pos_samples(pos_num) + self._get_neg_samples(neg_num)

        return batch

    def __len__(self):
        return self.length
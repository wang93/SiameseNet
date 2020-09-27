# encoding: utf-8
import torch
classes_num = 0
indices = [[]]
braid_indices = [[]]
batch_size = None


def parse_target(target):
    """each element of target ranges from 0. to (classes_num-1)"""
    global classes_num, indices, braid_indices, batch_size

    if isinstance(target, list):
        pass
    elif isinstance(target, torch.Tensor):
        target = target.view(-1).numpy().tolist()
    else:
        raise TypeError

    indices = [[] for _ in range(classes_num)]
    for i, e in enumerate(target):
        indices[int(e)].append(i)

    batch_size = len(target)
    braid_indices = []
    for sub_indices in indices:
        braid_sub_indices = sub_indices + [i + batch_size for i in sub_indices]
        braid_indices.append(braid_sub_indices)


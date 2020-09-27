# encoding: utf-8

classes_num = 0
indices = [[]]
braid_indices = [[]]
batch_size = None


def parse_target(target: list):
    """each element of target target range from 0. to (classes_num-1)"""
    global classes_num, indices, braid_indices, batch_size

    indices = [[] for _ in range(classes_num)]
    for i, e in enumerate(target):
        indices[int(e)].append(i)

    batch_size = len(target)
    braid_indices = []
    for sub_indices in indices:
        braid_sub_indices = sub_indices + [i + batch_size for i in sub_indices]
        braid_indices.append(braid_sub_indices)

    ####################
    # indices_0 = []
    # indices_1 = []
    # for i, e in enumerate(target):
    #     if e == 0.:
    #         indices_0.append(i)
    #     elif e == 1.:
    #         indices_1.append(i)
    #     else:
    #         raise ValueError
    #
    # batch_size = len(target)
    # braid_indices_0 = indices_0 + [i + batch_size for i in indices_0]
    # braid_indices_1 = indices_1 + [i + batch_size for i in indices_1]
    #
    # indices = [indices_0, indices_1]
    # braid_indices = [braid_indices_0, braid_indices_1]


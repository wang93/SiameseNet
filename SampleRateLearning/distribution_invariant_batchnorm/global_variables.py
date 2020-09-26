# encoding: utf-8

indices = [[]]
braid_indices = [[]]
batch_size = None


def parse_target(target: list):
    global indices, braid_indices, batch_size
    indices_0 = []
    indices_1 = []
    for i, e in enumerate(target):
        if e == 0.:
            indices_0.append(i)
        elif e == 1.:
            indices_1.append(i)
        else:
            raise ValueError

    batch_size = len(target)
    braid_indices_0 = indices_0 + [i + batch_size for i in indices_0]
    braid_indices_1 = indices_1 + [i + batch_size for i in indices_1]

    indices = [indices_0, indices_1]
    braid_indices = [braid_indices_0, braid_indices_1]


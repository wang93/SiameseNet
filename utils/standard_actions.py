# encoding: utf-8
import os
import random
import subprocess
import sys

import numpy as np
import torch

from config import opt
from utils.serialization import Logger


def _random_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def prepare_running(**kwargs):
    opt.parse_(kwargs)

    if not torch.cuda.is_available():
        raise NotImplementedError('This project must be implemented with CUDA!')

    sys.stdout = Logger(os.path.join(opt.exp_dir, 'log_train.txt'))
    print('current commit hash: {}'.format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))
    opt.print_()
    _random_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    return opt

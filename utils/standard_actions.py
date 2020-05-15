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


# def prepare_running(func):
#     def wrapper(**kwargs):
#         opt.parse_(kwargs)
#
#         os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opt.gpus])
#
#         if not torch.cuda.is_available():
#             raise NotImplementedError('This project must be implemented with CUDA!')
#
#         sys.stdout = Logger(os.path.join(opt.exp_dir, 'log_train.txt'))
#         print('current commit hash: {}'.format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))
#         opt.print_()
#         _random_seed(opt.seed)
#         torch.backends.cudnn.benchmark = True
#
#         func_return = func(opt)
#
#         return func_return
#
#     return wrapper

def prepare_running(**kwargs):
    opt.parse_(kwargs)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in opt.gpus])

    if not torch.cuda.is_available():
        raise NotImplementedError('This project must be implemented with CUDA!')

    sys.stdout = Logger(os.path.join(opt.exp_dir, 'log_train.txt'))
    print('current commit hash: {}'.format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))
    opt.print_()
    _random_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    return opt



def print_time(func):
    import time

    def wrapper(*args, **kwargs):
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        func_return = func(*args, **kwargs)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        return func_return

    return wrapper

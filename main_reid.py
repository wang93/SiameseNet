# encoding: utf-8
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

from config import opt
from primary_objects_factory import *
from utils.serialization import Logger


def random_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  #cudnn


def train(**kwargs):
    if not torch.cuda.is_available():
        raise NotImplementedError('This project must be implemented with CUDA!')
    opt.parse(kwargs)
    sys.stdout = Logger(os.path.join(opt.exp_dir, 'log_train.txt'))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('current commit hash: {}'.format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))
    opt.print()
    random_seed(opt.seed)
    torch.backends.cudnn.benchmark = True

    model, optimizer, start_epoch, best_rank1, best_epoch = get_model_with_optimizer(opt)
    data_loaders = get_dataloaders(opt, model.module.meta)
    reid_evaluator = get_evaluator(opt, model, **data_loaders)
    reid_trainer = get_trainer(opt, reid_evaluator, optimizer, best_rank1, best_epoch)

    if opt.evaluate:
        reid_evaluator.evaluate(re_ranking=opt.re_ranking, savefig=opt.savefig)
        reid_evaluator.evaluate(re_ranking=opt.re_ranking, savefig=opt.savefig, eval_flip=True)
        return

    for epoch in range(start_epoch+1, opt.max_epoch+1):
        reid_trainer.train(epoch, data_loaders['trainloader'])

    print('Best rank-1 {:.1%}, achieved at epoch {}'.format(reid_trainer.best_rank1, reid_trainer.best_epoch))
    reid_evaluator.evaluate(re_ranking=opt.re_ranking, eval_flip=True, savefig=os.path.join(opt.savefig, 'fused'))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    import fire
    fire.Fire()

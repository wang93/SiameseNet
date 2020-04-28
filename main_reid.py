# encoding: utf-8
import os
import random
import subprocess
import sys
import time
from pprint import pprint

import numpy as np
import torch
from torch.backends import cudnn

from config import opt
from primary_objects_factory import *
from utils.serialization import Logger


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


def random_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  #cudnn


def train(**kwargs):
    sys.stdout = Logger(os.path.join(opt.exp_dir, 'log_train.txt'))
    if not torch.cuda.is_available():
        raise NotImplementedError('This project must be implemented with CUDA!')
    opt._parse(kwargs)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('current commit hash: {}'.format(get_git_revision_hash()))
    print('=========experiment config==========')
    pprint(opt._state_dict())
    print('===============end==================')
    # set random seed and cudnn benchmark
    random_seed(opt.seed)
    cudnn.benchmark = True
    os.makedirs(opt.exp_dir, exist_ok=True)

    model, optimizer, start_epoch, best_rank1, best_epoch = get_model_with_optimizer(opt)

    trainloader, queryloader, galleryloader, queryFliploader, galleryFliploader\
        = get_dataloaders(opt, model.module.meta)

    reid_evaluator = get_evaluator(opt, model,
                                   queryloader=queryloader,
                                   galleryloader=galleryloader,
                                   queryFliploader=queryFliploader,
                                   galleryFliploader=galleryFliploader,
                                   minors_num=opt.eval_minors_num)

    reid_trainer = get_trainer(opt, reid_evaluator, optimizer, best_rank1, best_epoch)

    lr_strategy = get_lr_strategy(opt, optimizer)

    if opt.evaluate:
        reid_evaluator.evaluate(re_ranking=opt.re_ranking,
                                savefig=opt.savefig)

        reid_evaluator.evaluate(re_ranking=opt.re_ranking,
                                savefig=opt.savefig,
                                eval_flip=True)
        return

    # start training
    for epoch in range(start_epoch, opt.max_epoch):
        epoch_from_1 = epoch + 1
        if epoch_from_1 == opt.freeze_pretrained_untill:
            print('no longer freeze pretrained params (if there were any pretrained params)!')
            model.module.unlable_pretrained()
            optimizer = model.module.get_optimizer(optim=opt.optim,
                                                   lr=opt.lr,
                                                   momentum=opt.momentum,
                                                   weight_decay=opt.weight_decay)
            reid_trainer.optimizer = optimizer

        if opt.adjust_lr:
            lr_strategy(epoch_from_1)

        reid_trainer.train(epoch_from_1, trainloader)

    print('Best rank-1 {:.1%}, achieved at epoch {}'
          .format(reid_trainer.best_rank1, reid_trainer.best_epoch))

    savefig = os.path.join(opt.savefig, 'fused')
    reid_evaluator.evaluate(re_ranking=opt.re_ranking, eval_flip=True, savefig=savefig)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    import fire
    fire.Fire()

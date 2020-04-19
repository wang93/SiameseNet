# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData, ImagePairData, PairLoader
from datasets.samplers import RandomIdentitySampler, PosNegPairSampler
from models.networks import ResNetBuilder, IDE, Resnet, BFE
from models.braidnet import BraidNet
from trainers.evaluator import ResNetEvaluator, BraidNetEvaluator
from trainers.trainer import binary_logisticTrainer, cls_tripletTrainer
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
import random
import subprocess


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


def random_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  #cudnn


def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    random_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))
    
    print('working on commit {}'.format(get_git_revision_short_hash()))
    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')
########################################################################
    print('initializing model ...')
    if opt.model_name == 'braidnet':
        model = BraidNet(bi=(64, 128), braid=(128, 128, 128, 128), fc=(1,))
    else:
        raise NotImplementedError


    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    model_meta = model.meta
    if use_gpu:
        model = nn.DataParallel(model).cuda()
#########################################################
    print('initializing dataset {}'.format(opt.dataset))
    if isinstance(opt.dataset, list):
        dataset = data_manager.init_united_datasets(names=opt.dataset, mode=opt.mode)
    else:
        dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)

    if opt.test_pids_num >= 0:
        dataset.subtest2train(opt.test_pids_num)

    dataset.print_summary()

    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    if opt.model_name == 'braidnet':
        train_sampler = PosNegPairSampler(dataset.train, opt.pos_rate)
        trainloader = PairLoader(
            ImagePairData(dataset.train, TrainTransform(opt.datatype, model_meta)),
            sampler=train_sampler,
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True
        )
    else:
        train_sampler = RandomIdentitySampler(dataset.train, opt.num_instances)
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.datatype, model_meta)),
            sampler=train_sampler,
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True
        )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, model_meta)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, model_meta)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    queryFliploader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, model_meta, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryFliploader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, model_meta, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    if opt.model_name == 'braidnet':
        reid_evaluator = BraidNetEvaluator(model, minors_num=opt.eval_minors_num)
    else:
        reid_evaluator = ResNetEvaluator(model, minors_num=opt.eval_minors_num)

    if opt.evaluate:
        reid_evaluator.evaluate(queryloader,
                                galleryloader,
                                queryFliploader,
                                galleryFliploader,
                                re_ranking=opt.re_ranking,
                                savefig=opt.savefig)

        reid_evaluator.evaluate(queryloader,
                                galleryloader,
                                queryFliploader, galleryFliploader,
                                re_ranking=opt.re_ranking,
                                savefig=opt.savefig,
                                eval_flip=True)
        return

    if opt.model_name == 'braidnet':
        criterion = nn.BCELoss(reduction='elementwise_mean')
    else:
        raise NotImplementedError

    # get optimizer

    # if opt.model_name == 'braidnet':
    #params_reg, params_noreg = model.get_optim_policy()
    # else:
    #     raise NotImplementedError
    #
    # if opt.optim == "sgd":
    #     optimizer = torch.optim.SGD([{'params': params_reg, 'weight_decay': opt.weight_decay},
    #                                  {'params': params_noreg, 'weight_decay': 0.}],
    #                                 lr=opt.lr, momentum=0.9)
    # else:
    #     optimizer = torch.optim.Adam([{'params': params_reg, 'weight_decay': opt.weight_decay},
    #                                   {'params': params_noreg, 'weight_decay': 0.}],
    #                                  lr=opt.lr, momentum=0.9)
    optimizer = model.get_optimizer(optim=opt.optim,
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    if opt.model_name == 'braidnet':
        reid_trainer = binary_logisticTrainer(opt, model, optimizer, criterion, summary_writer, opt.correct_grads)
    else:
        reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion, summary_writer)

    def adjust_lr(optimizer, ep):
        # if ep < 50:
        #     lr = 1e-4*(ep//5+1)
        # else:
        #     lr = 1e-3 * (0.1 ** ((ep - 50) // 200))
        if ep < 100:
            lr = opt.lr
        else:
            lr = opt.lr * (opt.gamma ** ((ep-100)//20+1))

        for p in optimizer.param_groups:
            p['lr'] = lr

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            adjust_lr(optimizer, epoch + 1)
        reid_trainer.train(epoch, trainloader, opt.iter_num_per_epoch)
        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            if opt.mode == 'class':
                rank1 = test(model, queryloader)
            else:
                rank1 = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1}, 
                is_best=is_best, save_dir=opt.save_dir, 
                filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achieved at epoch {}'.format(best_rank1, best_epoch))
    reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader, eval_flip=True)


def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu() 
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1 


if __name__ == '__main__':
    import fire
    fire.Fire()

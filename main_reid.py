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
from datasets.data_loader import ImageData#, ImagePairData#, PairLoader
from datasets.samplers import RandomIdentitySampler, PosNegPairSampler
#from models.networks import ResNetBuilder, IDE, Resnet, BFE
from models.braidnet import BraidNet
from models.braidnet.braidmgn import BraidMGN
from trainers.evaluator import ResNetEvaluator, BraidNetEvaluator
from trainers.trainer import braidTrainer, cls_tripletTrainer
#from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin
#from utils.LiftedStructure import LiftedStructureLoss
#from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint, parse_checkpoints
from utils.transforms import TestTransform, TrainTransform
import random
import subprocess
import time


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

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('current commit hash: {}'.format(get_git_revision_hash()))
    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        raise NotImplementedError
########################################################################
    print('initializing model and optimizer...')
    if opt.model_name == 'braidnet':
        model = BraidNet(bi=(64, 128), braid=(128, 128, 128, 128), fc=(1,))
    elif opt.model_name == 'braidmgn':
        model = BraidMGN(feats=256, fc=(1,))
    else:
        raise NotImplementedError

    if opt.model_name == 'braidnet' and opt.pretrained_subparams:
        print('use pretrained params')
        model.load_pretrained()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    start_epoch = 0
    best_rank1 = -np.inf
    best_epoch = 0
    optimizer_state_dict = None
    if not opt.disable_resume:
        start_epoch, state_dict, best_epoch, best_rank1, optimizer_state_dict = parse_checkpoints(opt.save_dir)
        if start_epoch > 0:
            print('resume from epoch {0}'.format(start_epoch))
            print('the highest current rank-1 score is {0:.1%}, which was achieved after epoch {1}'.format(best_rank1, best_epoch))
            model.load_state_dict(state_dict, True)

    if opt.pretrained_subparams and start_epoch + 1 >= opt.freeze_pretrained_untill:
        print('no longer freeze pretrained params!')
        model.unlable_pretrained()

    model_meta = model.meta
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    else:
        raise NotImplementedError

    # get optimizer
    optimizer = model.module.get_optimizer(optim=opt.optim,
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay)

    if optimizer_state_dict is not None:
        print('optimizer comes to the state after epoch {0}'.format(start_epoch))
        optimizer.load_state_dict(optimizer_state_dict)

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

    if opt.model_name in ('braidnet', 'braidmgn'):
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=opt.augmentation)),
            sampler=PosNegPairSampler(data_source=dataset.train,
                                      pos_rate=opt.pos_rate,
                                      sample_num_per_epoch=opt.iter_num_per_epoch*opt.train_batch),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=False
        )
        #print('the length of trainloader is {0}'.format(len(trainloader)))
    else:
        raise NotImplementedError
        # trainloader = DataLoader(
        #     ImageData(dataset.train, TrainTransform(opt.datatype, model_meta)),
        #     sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
        #     batch_size=opt.train_batch, num_workers=opt.workers,
        #     pin_memory=pin_memory, drop_last=False
        # )

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

    if opt.model_name in ('braidnet', 'braidmgn'):
        criterion = nn.BCELoss(reduction='mean')
    else:
        raise NotImplementedError

    # get trainer
    if opt.model_name in ('braidnet', 'braidmgn'):
        reid_trainer = braidTrainer(opt, model, optimizer, criterion, summary_writer)
    else:
        raise NotImplementedError
        #reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion, summary_writer)

    def adjust_lr(optimizer, ep_from_1):
        #ep starts with 0
        ep_from_0 = ep_from_1 - 1
        if ep_from_0 < 100:
            mul = 1.
        else:
            mul = opt.gamma ** ((ep_from_0-100)//20+1)

        for p in optimizer.param_groups:
            p['lr'] = p['initial_lr'] * mul
            #print('in this param group, the base_lr is {0}, the lr is {1}'.format(p['base_lr'],p['lr'] ))

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
            adjust_lr(optimizer, epoch_from_1)

        reid_trainer.train(epoch_from_1, trainloader)
        # skip if not save model
        if opt.eval_step > 0 and epoch_from_1 % opt.eval_step == 0 or epoch_from_1 == opt.max_epoch:
            # if opt.mode == 'class':
            #     rank1 = test(model, queryloader)
            # else:
            savefig = os.path.join(opt.savefig, 'origin') if epoch_from_1 == opt.max_epoch else None
            rank1 = reid_evaluator.evaluate(queryloader,
                                            galleryloader,
                                            queryFliploader,
                                            galleryFliploader,
                                            savefig=savefig)

            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch_from_1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()

            save_checkpoint({'state_dict': state_dict, 'epoch': epoch_from_1, 'rank1': rank1},
                            is_best=is_best, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(epoch_from_1) + '.pth.tar')

            save_checkpoint({'state_dict': optimizer_state_dict, 'epoch': epoch_from_1},
                            is_best=False, save_dir=opt.save_dir,
                            filename='optimizer_checkpoint_ep' + str(epoch_from_1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achieved at epoch {}'.format(best_rank1, best_epoch))

    savefig = os.path.join(opt.savefig, 'fused')
    reid_evaluator.evaluate(queryloader,
                            galleryloader,
                            queryFliploader,
                            galleryFliploader,
                            eval_flip=True,
                            savefig=savefig)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


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

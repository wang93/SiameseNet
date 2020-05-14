# encoding: utf-8
import os
import os.path as osp
import re
import sys

import numpy as np
import torch

PREFIX_MODEL = 'model_checkpoint'
PREFIX_OPTIMIZER = 'optimizer_checkpoint'
BEST_MODEL_NAME = 'model_best.pth.tar'
CHECKPOINT_DIR = 'checkpoints'


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def save_checkpoint(state, exp_dir, epoch, prefix: str, eval_step=10):
    save_dir = osp.join(exp_dir, CHECKPOINT_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # delete previous checkpoints
    if epoch > 1 and (epoch - 1) % eval_step != 0:
        previous_file_name = '{0}_ep{1}.pth.tar'.format(prefix, epoch - 1)
        os.remove(osp.join(save_dir, previous_file_name))

    # save current checkpoint
    filename = '{0}_ep{1}.pth.tar'.format(prefix, epoch)
    fpath = osp.join(save_dir, filename)
    torch.save(state, fpath)


def save_current_status(model, optimizer, exp_dir, epoch, eval_step):
    model_state_dict = model.module.state_dict()
    optimizer_state_dict = optimizer.state_dict()

    save_checkpoint({'state_dict': model_state_dict, 'epoch': epoch},
                    exp_dir=exp_dir, epoch=epoch,
                    prefix=PREFIX_MODEL, eval_step=eval_step)

    save_checkpoint({'state_dict': optimizer_state_dict, 'epoch': epoch},
                    exp_dir=exp_dir, epoch=epoch,
                    prefix=PREFIX_OPTIMIZER, eval_step=eval_step)


def save_best_model(model, exp_dir, epoch, rank1):
    save_dir = osp.join(exp_dir, CHECKPOINT_DIR)
    os.makedirs(save_dir, exist_ok=True)
    fpath = osp.join(save_dir, BEST_MODEL_NAME)

    model_state_dict = model.module.state_dict()
    state = {'state_dict': model_state_dict, 'epoch': epoch, 'rank1': rank1}

    torch.save(state, fpath)


def get_best_model(exp_dir):
    f_path = osp.join(exp_dir, CHECKPOINT_DIR, BEST_MODEL_NAME)
    if os.path.exists(f_path):
        best_params = torch.load(f_path)
        best_state_dict = best_params['state_dict']
        best_epoch = best_params['epoch']
        best_rank1 = best_params['rank1']
    else:
        best_state_dict = None
        best_epoch = -1
        best_rank1 = 0

    return best_state_dict, best_epoch, best_rank1


def parse_checkpoints(exp_dir):
    load_dir = osp.join(exp_dir, CHECKPOINT_DIR)
    os.makedirs(load_dir, exist_ok=True)

    files = os.listdir(load_dir)
    files = [f for f in files if '.pth.tar' in f]
    if BEST_MODEL_NAME in files:
        files.remove(BEST_MODEL_NAME)
    pattern = re.compile(r'(?<=^{0}_ep)\d+'.format(PREFIX_MODEL))  # look for numbers
    epochs = [pattern.findall(f) for f in files]
    epochs = [int(e[0]) for e in epochs if len(e) > 0]

    start_epoch = 0
    state_dict = None
    best_rank1 = -np.inf
    best_epoch = 0
    if len(epochs) > 0:
        start_epoch = max(epochs)
        params_file_name = '{0}_ep{1}.pth.tar'.format(PREFIX_MODEL, start_epoch)
        params_file_path = osp.join(load_dir, params_file_name)
        state_dict = torch.load(params_file_path)['state_dict']

        best_params_file_path = osp.join(load_dir, BEST_MODEL_NAME)
        if os.path.exists(best_params_file_path):
            best_params = torch.load(best_params_file_path)
            best_rank1 = best_params['rank1']
            best_epoch = best_params['epoch']

    optimizer_state_dict = None
    if start_epoch > 0:
        optimizer_state_dict_path = os.path.join(load_dir, '{0}_ep{1}.pth.tar'.format(PREFIX_OPTIMIZER, start_epoch))

        if os.path.exists(optimizer_state_dict_path):
            optimizer_state_dict = torch.load(optimizer_state_dict_path)['state_dict']

    return start_epoch, state_dict, best_epoch, best_rank1, optimizer_state_dict



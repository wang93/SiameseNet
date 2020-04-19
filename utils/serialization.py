# encoding: utf-8
import errno
import os
import shutil
import sys

import os.path as osp
import torch

import re


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
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


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, is_best, save_dir, filename):
    fpath = osp.join(save_dir, filename)
    mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(save_dir, 'model_best.pth.tar'))


def find_latest_checkpoint(load_dir):
    files = os.listdir(load_dir)
    files = [f for f in files if '.pth.tar' in f]
    files.remove('model_best.pth.tar')
    pattern = re.compile(r'\d+') # look for numbers
    epochs = [pattern.findall(f) for f in files]
    epochs = [int(e[0]) for e in epochs]
    if len(epochs) > 0:
        start_epoch = max(epochs)
        start_epoch_index = epochs.index(start_epoch)
        params_file_name = files[start_epoch_index]
        params_file_path = osp.join(load_dir, params_file_name)
    else:
        start_epoch = 0
        params_file_path = None

    return start_epoch, params_file_path



from Utils.serialization import save_checkpoint, CHECKPOINT_DIR
from .loss import SRL_BCELoss

import os
import os.path as osp
import re

import torch

PREFIX_SRL = 'srl_checkpoint'


def save_current_srl_status(srl_loss: SRL_BCELoss, exp_dir, epoch, eval_step):
    srl_state_dict = srl_loss.state_dict()
    save_checkpoint({'state_dict': srl_state_dict, 'epoch': epoch},
                    exp_dir=exp_dir, epoch=epoch,
                    prefix=PREFIX_SRL, eval_step=eval_step)


def parse_srl_checkpoints(exp_dir):
    load_dir = osp.join(exp_dir, CHECKPOINT_DIR)
    os.makedirs(load_dir, exist_ok=True)

    files = os.listdir(load_dir)
    files = [f for f in files if '.pth.tar' in f and PREFIX_SRL in f]

    pattern = re.compile(r'(?<=^{0}_ep)\d+'.format(PREFIX_SRL))  # look for numbers
    epochs = [pattern.findall(f) for f in files]
    epochs = [int(e[0]) for e in epochs if len(e) > 0]

    start_epoch = 0
    srl_state_dict = None
    if len(epochs) > 0:
        start_epoch = max(epochs)
        params_file_name = '{0}_ep{1}.pth.tar'.format(PREFIX_SRL, start_epoch)
        params_file_path = osp.join(load_dir, params_file_name)
        srl_state_dict = torch.load(params_file_path)['state_dict']

    return start_epoch, srl_state_dict

import numpy as np
import torch
import torch.nn as nn

from utils.serialization import parse_checkpoints

__all__ = ['get_model_with_optimizer', ]


def get_model_with_optimizer(opt):
    print('initializing {0} model and its optimizer...'.format(opt.model_name))
    if opt.model_name == 'braidnet':
        from models.braidnet import BraidNet
        model = BraidNet(bi=(64, 128), braid=(128, 128, 128, 128), fc=(1,))
    elif opt.model_name == 'braidmgn':
        from models.braidnet.braidmgn import BraidMGN
        model = BraidMGN(feats=256, fc=(1,))
    else:
        raise NotImplementedError

    if opt.pretrained_subparams:
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
        start_epoch, state_dict, best_epoch, best_rank1, optimizer_state_dict = parse_checkpoints(opt.exp_dir)
        if start_epoch > 0:
            print('resume from epoch {0}'.format(start_epoch))
            print('the highest current rank-1 score is {0:.1%}, which was achieved after epoch {1}'.format(best_rank1, best_epoch))
            model.load_state_dict(state_dict, True)

    if opt.pretrained_subparams and start_epoch + 1 >= opt.freeze_pretrained_untill:
        print('no longer freeze pretrained params!')
        model.unlable_pretrained()

    #model_meta = model.meta

    model = nn.DataParallel(model).cuda()

    if opt.sync_bn:
        from sync_batchnorm.batchnorm import convert_model
        model = convert_model(model)

    # get optimizer
    optimizer = model.module.get_optimizer(optim=opt.optim,
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay)

    if optimizer_state_dict is not None:
        print('optimizer comes to the state after epoch {0}'.format(start_epoch))
        optimizer.load_state_dict(optimizer_state_dict)

    return model, optimizer, start_epoch, best_rank1, best_epoch

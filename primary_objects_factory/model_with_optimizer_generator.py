import torch
from torch import nn

from utils.data_parallel import DataParallel
from utils.serialization import parse_checkpoints

__all__ = ['get_model_with_optimizer', ]


def get_model_with_optimizer(opt, id_num=1, naive=False):
    if not naive:
        print('initializing model {0} and its optimizer...'.format(opt.model_name))

    if opt.loss == 'bce':
        fc = (opt.tail_times,)
        score2prob = lambda x: x.mean(dim=1)
        # score2prob = lambda x: nn.Sigmoid()(x).mean(dim=1)

    elif opt.loss == 'triplet':
        fc = (1,)
        score2prob = lambda x: x.mean(dim=1)

    elif opt.loss == 'ce':
        fc = (2,)
        score2prob = lambda x: nn.Softmax(dim=1)(x)[:, 1]

    else:
        raise NotImplementedError

    if not naive:
        print('the setting of fc layers is {0}'.format(fc))

    if opt.model_name == 'braidnet':
        from models.braidnet import BraidNet
        model = BraidNet(bi=(64, 128), braid=(128, 128, 128, 128), fc=fc, score2prob=score2prob)

    elif opt.model_name == 'braidmgn':
        from models.braidnet.braidmgn import BraidMGN
        model = BraidMGN(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'mmbraidmgn':
        from models.braidnet.braidmgn import MMBraidMGN
        model = MMBraidMGN(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'densebraidmgn':
        from models.braidnet.braidmgn import DenseBraidMGN
        model = DenseBraidMGN(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'resbraidmgn':
        from models.braidnet.braidmgn import ResBraidMGN
        model = ResBraidMGN(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'osnet':
        from models.braidnet.braidosnet import OSNet
        model = OSNet(feats=opt.feats, num_classes=id_num)

    else:
        raise NotImplementedError

    if naive:
        return model

    if opt.pretrained_subparams:
        print('use pretrained params')
        model.load_pretrained()

    if opt.zero_tail_weight:
        print('weights in the final fc layer are initialized to zero')
        model.zero_tail_weight()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)

    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    start_epoch = 0
    # best_rank1 = -np.inf
    # best_epoch = 0
    optimizer_state_dict = None
    if not opt.disable_resume:
        start_epoch, state_dict, best_epoch, best_rank1, optimizer_state_dict = parse_checkpoints(opt.exp_dir)
        if best_epoch > 0:
            print('the highest current rank-1 score is {0:.1%}, which was achieved after epoch {1}'.format(best_rank1,
                                                                                                           best_epoch))
        if start_epoch > 0:
            print('net comes to the state after epoch {0}'.format(start_epoch))
            model.load_state_dict(state_dict, True)

    if start_epoch + 1 >= opt.freeze_pretrained_untill:
        print('no longer freeze pretrained params (if there are any pretrained params)')
        model.unlable_pretrained()

    model = DataParallel(model).cuda()  # nn.DataParallel(model).cuda()

    if opt.sync_bn:
        from utils.sync_batchnorm.batchnorm import convert_model
        model = convert_model(model)

    # get optimizer
    optimizer = model.module.get_optimizer(optim=opt.optim,
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay)

    if optimizer_state_dict is not None:
        print('optimizer comes to the state after epoch {0}'.format(start_epoch))
        optimizer.load_state_dict(optimizer_state_dict)

    return model, optimizer, start_epoch  # , best_rank1, best_epoch

import torch
from torch import nn

from utils.data_parallel import DataParallel
from utils.serialization import parse_checkpoints

__all__ = ['get_model_with_optimizer', ]


def get_model_with_optimizer(opt, id_num=1, naive=False):
    if not naive:
        print('initializing model {0} and its optimizer...'.format(opt.model_name))

    if opt.loss in ['bce', 'lbce']:
        fc = (opt.tail_times,)
        score2prob = lambda x: x.mean(dim=1)
        # score2prob = lambda x: nn.Sigmoid()(x).mean(dim=1)

    elif opt.loss == 'triplet':
        fc = (1,)
        score2prob = lambda x: - x.mean(dim=1)

    elif opt.loss == 'ce':
        fc = (2,)
        score2prob = lambda x: nn.Softmax(dim=1)(x)[:, 1]

    elif opt.loss == 'lsce':
        fc = None
        score2prob = None

    elif opt.loss == 'lsce_bce':
        fc = (opt.tail_times,)
        score2prob = lambda x: x.mean(dim=1)

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

    # elif opt.model_name == 'eulideanosnet':
    #     from models.braidnet.braidosnet import EulideanOSNet
    #     model = EulideanOSNet(feats=opt.feats, num_classes=id_num)

    elif opt.model_name == 'squareosnet':
        from models.braidnet.braidosnet import SquareOSNet
        model = SquareOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'bbosnet':
        from models.braidnet.braidosnet import BBOSNet
        model = BBOSNet(feats=opt.feats,
                        fc=fc,
                        num_classes=id_num,
                        score2prob=score2prob)

    elif opt.model_name == 'bbmosnet':
        from models.braidnet.braidosnet import BBMOSNet
        model = BBMOSNet(feats=opt.feats,
                         fc=fc,
                         num_classes=id_num,
                         score2prob=score2prob)

    elif opt.model_name == 'wbbmosnet':
        from models.braidnet.braidosnet import WBBMOSNet
        model = WBBMOSNet(feats=opt.feats,
                          fc=fc,
                          num_classes=id_num,
                          score2prob=score2prob)

    elif opt.model_name == 'sumsquareosnet':
        from models.braidnet.braidosnet import SumSquareOSNet
        model = SumSquareOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'meansquareosnet':
        from models.braidnet.braidosnet import MeanSquareOSNet
        model = MeanSquareOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'aabraidosnet':
        from models.braidnet.braidosnet import AABraidOSNet
        model = AABraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'aaabraidosnet':
        from models.braidnet.braidosnet import AAABraidOSNet
        model = AAABraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'aaasbraidosnet':
        from models.braidnet.braidosnet import AAASBraidOSNet
        model = AAASBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'aaboss':
        from models.braidnet.braidosnet import AABOSS
        model = AABOSS(feats=opt.feats, w_num=opt.w_num, fc=fc, score2prob=score2prob, num_classes=id_num)

    elif opt.model_name == 'wbboss':
        from models.braidnet.braidosnet import WBBOSS
        model = WBBOSS(feats=opt.feats,
                       fc=fc,
                       num_classes=id_num,
                       score2prob=score2prob)

    elif opt.model_name == 'aa2braidosnet':
        from models.braidnet.braidosnet import AA2BraidOSNet
        model = AA2BraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'aa3braidosnet':
        from models.braidnet.braidosnet import AA3BraidOSNet
        model = AA3BraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'aa4braidosnet':
        from models.braidnet.braidosnet import AA4BraidOSNet
        model = AA4BraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'minmaxosnet':
        from models.braidnet.braidosnet import MinMaxOSNet
        model = MinMaxOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'minwmaxybraidosnet':
        from models.braidnet.braidosnet import MinWMaxYBraidOSNet
        model = MinWMaxYBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'minwmmybraidosnet':
        from models.braidnet.braidosnet import MinWMMYBraidOSNet
        model = MinWMMYBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'min2wmmybraidosnet':
        from models.braidnet.braidosnet import Min2WMMYBraidOSNet
        model = Min2WMMYBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'minbnwmmybraidosnet':
        from models.braidnet.braidosnet import MinBNWMMYBraidOSNet
        model = MinBNWMMYBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'minbn2wmmybraidosnet':
        from models.braidnet.braidosnet import MinBN2WMMYBraidOSNet
        model = MinBN2WMMYBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'mmbraidosnet':
        from models.braidnet.braidosnet import MMBraidOSNet
        model = MMBraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

    elif opt.model_name == 'braidosnet':
        from models.braidnet.braidosnet import BraidOSNet
        model = BraidOSNet(feats=opt.feats, fc=fc, score2prob=score2prob)

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

    if opt.sync_bn:
        from utils.sync_batchnorm.batchnorm import convert_model
        model = convert_model(model)

    elif opt.stable_bn0:
        print('BN layers in Braid & FC structures are in stable version 0.')
        from SampleRateLearning.stable_batchnorm.batchnorm0 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn1:
        print('BN layers in Braid & FC structures are in stable version 1.')
        from SampleRateLearning.stable_batchnorm.batchnorm1 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn2:
        print('BN layers in Braid & FC structures are in stable version 2.')
        from SampleRateLearning.stable_batchnorm.batchnorm2 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn3:
        print('BN layers in Braid & FC structures are in stable version 3.')
        from SampleRateLearning.stable_batchnorm.batchnorm3 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn4:
        print('BN layers in Braid & FC structures are in stable version 4, which uses bias-corrected '
              'running mean & var all the time.')
        from SampleRateLearning.stable_batchnorm.batchnorm4 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn5:
        print('BN layers in Braid & FC structures are in stable version 5, which /max(mean(stds), '
              'sqrt(eps)) and uses bias-corrected running mean & std all the time.')
        from SampleRateLearning.stable_batchnorm.batchnorm5 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn6:
        print('BN layers in Braid & FC structures are in stable version 6.')
        from SampleRateLearning.stable_batchnorm.batchnorm6 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn7:
        print('BN layers in Braid & FC structures are in stable version 7.')
        from SampleRateLearning.stable_batchnorm.batchnorm7 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn8:
        print('BN layers in Braid & FC structures are in stable version 8.')
        from SampleRateLearning.stable_batchnorm.batchnorm8 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn9:
        print('BN layers in Braid & FC structures are in stable version 9.')
        from SampleRateLearning.stable_batchnorm.batchnorm9 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn10:
        print('BN layers in Braid & FC structures are in stable version 10.')
        from SampleRateLearning.stable_batchnorm.batchnorm10 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn11:
        print('BN layers in Braid & FC structures are in stable version 11.')
        from SampleRateLearning.stable_batchnorm.batchnorm11 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn12:
        print('BN layers in Braid & FC structures are in stable version 12.')
        from SampleRateLearning.stable_batchnorm.batchnorm12 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn13:
        print('BN layers in Braid & FC structures are in stable version 13.')
        from SampleRateLearning.stable_batchnorm.batchnorm13 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn14:
        print('BN layers in Braid & FC structures are in stable version 14.')
        from SampleRateLearning.stable_batchnorm.batchnorm14 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn15:
        print('BN layers in Braid & FC structures are in stable version 15.')
        from SampleRateLearning.stable_batchnorm.batchnorm15 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn16:
        print('BN layers in Braid & FC structures are in stable version 16.')
        from SampleRateLearning.stable_batchnorm.batchnorm16 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn17:
        print('BN layers in Braid & FC structures are in stable version 17.')
        from SampleRateLearning.stable_batchnorm.batchnorm17 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    elif opt.stable_bn18:
        print('BN layers in Braid & FC structures are in stable version 18.')
        from SampleRateLearning.stable_batchnorm.batchnorm18 import convert_model
        model.braid = convert_model(model.braid)
        model.fc = convert_model(model.fc)

    if opt.wc:
        print('incorporate weight centralization (WC).')
        from WeightModification.centralization import convert_model as convert_model_wc
        model = convert_model_wc(model)

    start_epoch = 0
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

    # get optimizer
    optimizer = model.module.get_optimizer(optim=opt.optim,
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay,
                                           gc=opt.gc,
                                           gc_loc=opt.gc_loc)

    if optimizer_state_dict is not None:
        print('optimizer comes to the state after epoch {0}'.format(start_epoch))
        optimizer.load_state_dict(optimizer_state_dict)

    return model, optimizer, start_epoch  # , best_rank1, best_epoch

from os.path import join as path_join

from tensorboardX import SummaryWriter

from .lr_strategy_generator import get_lr_strategy


def get_trainer(opt, evaluator, optimizer, best_rank1, best_epoch):

    summary_writer = SummaryWriter(path_join(opt.exp_dir, 'tensorboard_log'))

    lr_strategy = get_lr_strategy(opt, optimizer)

    if opt.loss == 'bce':
        from torch.nn import BCELoss
        criterion = BCELoss()

    elif opt.loss == 'triplet':
        from utils.loss import TripletLoss4Braid
        criterion = TripletLoss4Braid(opt.margin)

    else:
        raise NotImplementedError

    if opt.loss == 'bce':
        from trainers.trainer import braidTrainer
        reid_trainer = braidTrainer(opt, evaluator, optimizer, lr_strategy, criterion,
                                    summary_writer, best_rank1, best_epoch)

    elif opt.loss == 'triplet':
        from trainers.trainer import braid_tripletTrainer
        reid_trainer = braid_tripletTrainer(opt, evaluator, optimizer, lr_strategy, criterion,
                                            summary_writer, best_rank1, best_epoch)

    else:
        raise NotImplementedError
        # reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion, summary_writer)

    return reid_trainer

from os.path import join as path_join

from tensorboardX import SummaryWriter

from .lr_strategy_generator import get_lr_strategy


def get_trainer(opt, evaluator, optimizer, best_rank1, best_epoch):

    summary_writer = SummaryWriter(path_join(opt.exp_dir, 'tensorboard_log'))

    lr_strategy = get_lr_strategy(opt)

    if opt.train_mode == 'pair':
        if opt.loss == 'bce':
            from torch.nn import BCELoss
            criterion = BCELoss()

        else:
            raise NotImplementedError

        from trainers.trainer import BraidPairTrainer
        reid_trainer = BraidPairTrainer(opt, evaluator, optimizer, lr_strategy, criterion,
                                        summary_writer, best_rank1, best_epoch, opt.train_phase_num)

    elif opt.train_mode == 'cross':
        if opt.loss == 'bce':
            from utils.loss import CrossSimilarityBCELoss
            criterion = CrossSimilarityBCELoss()

        elif opt.loss == 'triplet':
            from utils.loss import TripletLoss4Braid
            criterion = TripletLoss4Braid(opt.margin)

        else:
            raise NotImplementedError

        from trainers.trainer import BraidCrossTrainer
        reid_trainer = BraidCrossTrainer(opt, evaluator, optimizer, lr_strategy, criterion,
                                         summary_writer, best_rank1, best_epoch, opt.train_phase_num)

    else:
        raise NotImplementedError

    return reid_trainer

# encoding: utf-8
import time
from os.path import join

from primary_objects_factory import *
from utils.standard_actions import prepare_running


def train(**kwargs):
    opt = prepare_running(**kwargs)

    model, optimizer, start_epoch, best_rank1, best_epoch = get_model_with_optimizer(opt)
    data_loaders = get_dataloaders(opt, model.module.meta)
    reid_evaluator = get_evaluator(opt, model, **data_loaders)
    reid_trainer = get_trainer(opt, reid_evaluator, optimizer, best_rank1, best_epoch)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if opt.evaluate:
        reid_evaluator.evaluate(re_ranking=opt.re_ranking, savefig=opt.savefig)
        reid_evaluator.evaluate(re_ranking=opt.re_ranking, savefig=opt.savefig, eval_flip=True)
        return

    for epoch in range(start_epoch + 1, opt.max_epoch + 1):
        reid_trainer.train(epoch, data_loaders['trainloader'])

    print('Best rank-1 {:.1%}, achieved at epoch {}'.format(reid_trainer.best_rank1, reid_trainer.best_epoch))
    reid_evaluator.evaluate(re_ranking=opt.re_ranking, eval_flip=True, savefig=join(opt.savefig, 'fused'))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    import fire
    fire.Fire()

# encoding: utf-8
import time

from primary_objects_factory import *
from utils.standard_actions import prepare_running


def train(**kwargs):
    opt = prepare_running(**kwargs)

    model, optimizer, done_epoch = get_model_with_optimizer(opt)
    data_loaders = get_dataloaders(opt, model.module.meta)
    reid_evaluator = get_evaluator(opt, model, **data_loaders)
    reid_trainer = get_trainer(opt, data_loaders['trainloader'], reid_evaluator, optimizer)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if opt.evaluate:
        reid_evaluator.evaluate(re_ranking=opt.re_ranking, savefig=opt.savefig)
        reid_evaluator.evaluate(re_ranking=opt.re_ranking, savefig=opt.savefig, eval_flip=True)

    reid_trainer.train_from(done_epoch)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    import fire
    fire.Fire()

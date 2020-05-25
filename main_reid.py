# encoding: utf-8
from primary_objects_factory import get_trainer
from utils.standard_actions import prepare_running


def train(**kwargs):
    opt = prepare_running(**kwargs)
    reid_trainer = get_trainer(opt)

    if opt.evaluate:
        reid_trainer.evaluate_best()
        # reid_trainer.evaluate()
        return

    if opt.check_discriminant:
        reid_trainer.check_discriminant()
        return

    reid_trainer.continue_train()


if __name__ == '__main__':
    import fire
    fire.Fire()

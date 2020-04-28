from torch.utils.data import DataLoader

from datasets import data_manager
from datasets.data_loader import ImageData
from utils.transforms import TestTransform, TrainTransform


def get_dataloaders(opt, model_meta):
    print('initializing {} dataset ...'.format(opt.dataset))

    if isinstance(opt.dataset, list):
        dataset = data_manager.init_united_datasets(names=opt.dataset, mode=opt.mode)
    else:
        dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)

    if opt.test_pids_num >= 0:
        dataset.subtest2train(opt.test_pids_num)

    dataset.print_summary()

    pin_memory = True

    if opt.model_name in ('braidnet', 'braidmgn'):
        from datasets.samplers import PosNegPairSampler
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=opt.augmentation)),
            sampler=PosNegPairSampler(data_source=dataset.train,
                                      pos_rate=opt.pos_rate,
                                      sample_num_per_epoch=opt.iter_num_per_epoch*opt.train_batch),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=False
        )
    else:
        raise NotImplementedError

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, model_meta)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, model_meta)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    queryFliploader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, model_meta, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryFliploader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, model_meta, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    return {'trainloader': trainloader,
            'queryloader':queryloader,
            'galleryloader': galleryloader,
            'queryFliploader': queryFliploader,
            'galleryFliploader': galleryFliploader}

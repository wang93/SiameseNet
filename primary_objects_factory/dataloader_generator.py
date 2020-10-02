from torch.utils.data import DataLoader

from dataset import data_info
from dataset.data_image import ImageData, PreLoadedImageData
from dataset.transforms import TestTransform, TrainTransform


def get_dataloaders(opt, model_meta):
    print('initializing {} dataset ...'.format(opt.dataset))

    # train_relabel = (opt.train_mode == 'normal' or opt.loss == 'lsce_bce') and (not opt.check_discriminant)
    train_relabel = not (opt.check_discriminant
                         or opt.check_element_discriminant
                         or opt.check_pair_effect
                         or opt.sort_pairs_by_scores)

    if train_relabel:
        print('note: the training set is relabeled!')

    if isinstance(opt.dataset, list):
        dataset = data_info.init_united_datasets(names=opt.dataset, train_relabel=train_relabel)
    else:
        dataset = data_info.init_dataset(name=opt.dataset, train_relabel=train_relabel)

    if opt.test_pids_num >= 0:
        dataset.subtest2train(opt.test_pids_num)

    if opt.eval_fast:
        dataset.reduce_query()

    dataset.print_summary()

    pin_memory = True

    if opt.check_discriminant or opt.check_element_discriminant or opt.check_pair_effect or opt.sort_pairs_by_scores:
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=None)),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory,
        )

        dataset.query.extend(dataset.gallery)

        queryloader = DataLoader(
            ImageData(dataset.query, TestTransform(opt.datatype, model_meta)),
            batch_size=opt.test_batch, num_workers=opt.workers,
            pin_memory=pin_memory
        )

        return {'trainloader': trainloader,
                'queryloader': queryloader,
                'galleryloader': None,
                'queryFliploader': None,
                'galleryFliploader': None}

    if opt.train_mode == 'normal':
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=opt.augmentation)),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True, shuffle=True
        )

    elif opt.train_mode == 'pair':
        if opt.srl:
            print('Sampler supports SRL!')
            from SampleRateLearning.sampler import SampleRateBatchSampler #SampleRateSampler
            batch_sampler = SampleRateBatchSampler(data_source=dataset.train,
                                                   sample_num_per_epoch=opt.iter_num_per_epoch * opt.train_batch,
                                                   batch_size=opt.train_batch)
            trainloader = DataLoader(
                PreLoadedImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=opt.augmentation)),
                batch_sampler=batch_sampler,
                num_workers=0,
                pin_memory=pin_memory,
            )
            print('num_workers=0 in the training loader.')

        else:
            from dataset.samplers import PosNegPairSampler
            sampler = PosNegPairSampler(data_source=dataset.train,
                                        pos_rate=opt.pos_rate,
                                        sample_num_per_epoch=opt.iter_num_per_epoch * opt.train_batch)

            trainloader = DataLoader(
                ImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=opt.augmentation)),
                sampler=sampler,
                batch_size=opt.train_batch, num_workers=opt.workers,
                pin_memory=pin_memory, drop_last=False
            )

    elif opt.train_mode in ['cross', 'ide_cross']:
        from dataset.samplers import RandomIdentitySampler
        trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(opt.datatype, model_meta, augmentaion=opt.augmentation)),
            sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True
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
            'queryloader': queryloader,
            'galleryloader': galleryloader,
            'queryFliploader': queryFliploader,
            'galleryFliploader': galleryFliploader}

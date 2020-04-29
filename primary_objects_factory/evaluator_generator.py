

def get_evaluator(opt, model, queryloader, galleryloader, queryFliploader, galleryFliploader, ranks=(1, 2, 4, 5, 8, 10, 16, 20), **kwargs):
    print('initializing evaluator...')

    if opt.eval_phase_num == 1:
        from trainers.evaluator import BraidEvaluator
        reid_evaluator = BraidEvaluator(model,
                                        queryloader=queryloader,
                                        galleryloader=galleryloader,
                                        queryFliploader=queryFliploader,
                                        galleryFliploader=galleryFliploader,
                                        minors_num=opt.eval_minors_num,
                                        ranks=ranks)
    elif opt.eval_phase_num == 2:
        from trainers.evaluator import BraidEvaluator_2Phases
        reid_evaluator = BraidEvaluator_2Phases(model,
                                                queryloader=queryloader,
                                                galleryloader=galleryloader,
                                                queryFliploader=queryFliploader,
                                                galleryFliploader=galleryFliploader,
                                                minors_num=opt.eval_minors_num,
                                                ranks=ranks)

    else:

        raise NotImplementedError

    return reid_evaluator


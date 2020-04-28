

def get_evaluator(opt, model, queryloader, galleryloader, queryFliploader, galleryFliploader, ranks=(1, 2, 4, 5, 8, 10, 16, 20), **kwargs):
    print('initializing evaluator...')

    if opt.model_name in ('braidnet', 'braidmgn'):
        from trainers.evaluator import BraidEvaluator
        reid_evaluator = BraidEvaluator(model,
                                        queryloader=queryloader,
                                        galleryloader=galleryloader,
                                        queryFliploader=queryFliploader,
                                        galleryFliploader=galleryFliploader,
                                        minors_num=opt.eval_minors_num,
                                        ranks=ranks)
    else:
        from trainers.evaluator import ResNetEvaluator
        reid_evaluator = ResNetEvaluator(model,
                                         queryloader=queryloader,
                                         galleryloader=galleryloader,
                                         queryFliploader=queryFliploader,
                                         galleryFliploader=galleryFliploader,
                                         minors_num=opt.eval_minors_num,
                                         ranks=ranks)

    return reid_evaluator


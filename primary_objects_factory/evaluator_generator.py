

def get_evaluator(opt, model, queryloader, galleryloader, queryFliploader, galleryFliploader, minors_num=0, ranks=(1, 2, 4, 5, 8, 10, 16, 20)):
    print('initializing evaluator...')

    if opt.model_name in ('braidnet', 'braidmgn'):
        from trainers.evaluator import BraidEvaluator
        reid_evaluator = BraidEvaluator(model,
                                        queryloader=queryloader,
                                        galleryloader=galleryloader,
                                        queryFliploader=queryFliploader,
                                        galleryFliploader=galleryFliploader,
                                        minors_num=opt.eval_minors_num)
    else:
        from trainers.evaluator import ResNetEvaluator
        reid_evaluator = ResNetEvaluator(model,
                                         queryloader=queryloader,
                                         galleryloader=galleryloader,
                                         queryFliploader=queryFliploader,
                                         galleryFliploader=galleryFliploader,
                                         minors_num=opt.eval_minors_num)

    return reid_evaluator


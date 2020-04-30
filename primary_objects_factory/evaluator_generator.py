

def get_evaluator(opt, model, queryloader, galleryloader, queryFliploader, galleryFliploader, ranks=(1, 2, 4, 5, 8, 10, 16, 20), **kwargs):
    print('initializing evaluator...')

    from trainers.evaluator import ReIDEvaluator
    reid_evaluator = ReIDEvaluator(model,
                                   queryloader=queryloader,
                                   galleryloader=galleryloader,
                                   queryFliploader=queryFliploader,
                                   galleryFliploader=galleryFliploader,
                                   phase_num=opt.eval_phase_num,
                                   minors_num=opt.eval_minors_num,
                                   ranks=ranks)

    return reid_evaluator

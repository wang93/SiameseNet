

def get_evaluator(opt, model, queryloader, galleryloader, queryFliploader, galleryFliploader, ranks=(1, 2, 4, 5, 8, 10, 16, 20), **kwargs):
    print('initializing evaluator...')

    from agents.evaluator import ReIDEvaluator

    reid_evaluator = ReIDEvaluator(model,
                                   opt,
                                   queryloader=queryloader,
                                   galleryloader=galleryloader,
                                   queryFliploader=queryFliploader,
                                   galleryFliploader=galleryFliploader,
                                   ranks=ranks)

    return reid_evaluator

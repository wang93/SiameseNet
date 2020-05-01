def MultiStepLR(optimizer, ep_from_1, gamma):
    ep_from_0 = ep_from_1 - 1
    if ep_from_0 < 100:
        mul = 1.
    else:
        mul = gamma ** ((ep_from_0 - 100) // 20 + 1)

    for p in optimizer.param_groups:
        p['lr'] = p['initial_lr'] * mul


def MultiStepLR_LinearWarmUp(optimizer, ep_from_1, gamma, warmup_till):
    ep_from_0 = ep_from_1 - 1
    warmup_till -= 1
    if ep_from_0 < warmup_till:
        mul = float(ep_from_0 + 1) / float(warmup_till + 1)
    elif ep_from_0 < 100:
        mul = 1.
    else:
        mul = gamma ** ((ep_from_0 - 100) // 20 + 1)

    for p in optimizer.param_groups:
        p['lr'] = p['initial_lr'] * mul


def get_lr_strategy(opt):
    if opt.warmup_till <= 1:
        return lambda o, e: MultiStepLR(o, e, opt.gamma)
    else:
        return lambda o, e: MultiStepLR_LinearWarmUp(o, e, opt.gamma, opt.warmup_till)

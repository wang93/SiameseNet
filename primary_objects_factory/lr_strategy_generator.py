

def MultiStepLR(optimizer, ep_from_1, gamma):
    # ep starts with 0
    ep_from_0 = ep_from_1 - 1
    if ep_from_0 < 100:
        mul = 1.
    else:
        mul = gamma ** ((ep_from_0 - 100) // 20 + 1)

    for p in optimizer.param_groups:
        p['lr'] = p['initial_lr'] * mul


def get_lr_strategy(opt):
    return lambda o, e: MultiStepLR(o, e, opt.gamma)

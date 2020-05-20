# def MultiStepLR(optimizer, ep_from_1, gamma):
#     ep_from_0 = ep_from_1 - 1
#     if ep_from_0 < 200:
#         mul = 1.
#     else:
#         mul = gamma ** ((ep_from_0 - 200) // 20 + 1)
#
#     for p in optimizer.param_groups:
#         p['lr'] = p['initial_lr'] * mul


def MultiStepLR_LinearWarmUp(optimizer, ep_from_1, gamma, warmup_till):
    ep_from_0 = ep_from_1 - 1
    warmup_till -= 1
    if ep_from_0 < warmup_till:
        mul = float(ep_from_0 + 1) / float(warmup_till + 1)
    elif ep_from_0 < 200:
        mul = 1.
    else:
        mul = gamma ** ((ep_from_0 - 200) // 20 + 1)

    for p in optimizer.param_groups:
        p['lr'] = p['initial_lr'] * mul


def MileStoneLR_LinearWarmUp(optimizer, ep_from_1, gamma, warmup_till, milsestones):
    ep_from_0 = ep_from_1 - 1
    warmup_till -= 1
    if ep_from_0 < warmup_till:
        mul = float(ep_from_0 + 1) / float(warmup_till + 1)

    else:
        i = 0
        for i, m in enumerate(milsestones):
            if ep_from_1 <= m:
                break

        mul = gamma ** i

    for p in optimizer.param_groups:
        p['lr'] = p['initial_lr'] * mul


def get_lr_strategy(opt):
    if len(opt.milestones) == 0:
        return lambda o, e: MultiStepLR_LinearWarmUp(o, e, opt.gamma, opt.warmup_till)
    else:
        return lambda o, e: MileStoneLR_LinearWarmUp(o, e, opt.gamma, opt.warmup_till, opt.milestones)

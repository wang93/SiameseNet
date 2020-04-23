# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'market1501'
    datatype = 'person'
    mode = 'retrieval'
    pos_rate = 0.5
    num_instances = 4

    # optimization options
    loss = 'bce'
    optim = 'sgd'
    max_epoch = 280
    iter_num_per_epoch = 500
    train_batch = 256
    test_batch = 256
    adjust_lr = False
    freeze_pretrained_untill = -1 #=0, 1, 2... =0 when always freeze pretrained
    lr = 0.4
    adjust_lr = False
    gamma = 0.5
    weight_decay = 5e-4
    momentum = 0.9
    #random_crop = False
    margin = None
    num_gpu = 1

    #evaluation options
    evaluate = False
    savefig = None 
    re_ranking = False
    eval_step = 50
    test_pids_num = 100  # = <0 when don't change test set
    eval_minors_num = 100  # <=0 when evaluation on the whole test set one time

    # model options
    model_name = 'braidnet'  # triplet, softmax_triplet, bfe, ide, braidnet
    last_stride = 1
    resnet_stem = False
    pretrained_model = None
    disable_resume = False
    
    # miscs
    print_freq = 30
    save_dir = './pytorch-ckpt/market'
    workers = 8

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt does not have attribute %s" % k)
            setattr(self, k, v)
        #print(self.dataset)

        if self.dataset[0] == '[':
            self.dataset = eval(self.dataset)
            #print(self.dataset)

        self.datatype = 'person'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()

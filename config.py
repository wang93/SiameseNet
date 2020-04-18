# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'market1501'
    datatype = 'person'
    mode = 'retrieval'
    test_pids_num = 100 # = <0 when don't change test set
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
    lr = 0.4
    adjust_lr = False
    gamma = 0.5
    weight_decay = 5e-4
    momentum = 0.9
    #random_crop = False
    margin = None
    num_gpu = 1
    correct_grads = False

    #evaluation options
    evaluate = False
    savefig = None 
    re_ranking = False
    eval_step = 50
    eval_minors_num = 100  # <=0 when evaluation on the whole test set one time

    # model options
    model_name = 'braidnet'  # triplet, softmax_triplet, bfe, ide, braidnet
    last_stride = 1
    pretrained_model = None
    
    # miscs
    print_freq = 30
    save_dir = './pytorch-ckpt/market'
    workers = 10
    start_epoch = 0
    best_rank = -np.inf


    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print(self.dataset)

        if self.dataset[0] == '[':
            self.dataset = eval(self.dataset)
            print(self.dataset)

        self.datatype = 'person'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()

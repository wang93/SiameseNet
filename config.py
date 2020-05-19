# encoding: utf-8
import warnings
from os.path import join
from pprint import pprint


class DefaultConfig(object):
    seed = 0
    gpus = [0, 1]

    # dataset options
    dataset = 'market1501'
    datatype = 'person'
    augmentation = None  # 'Cutout' or 'RandomErasing' or None
    mode = 'retrieval'
    pos_rate = 0.5
    num_instances = 4
    workers = 8

    # optimization options
    loss = 'bce'  # bce / triplet / ce / lsce
    optim = 'sgd'
    max_epoch = 280
    iter_num_per_epoch = 500
    train_batch = 256
    train_phase_num = 1  # 1 / 2
    train_mode = 'pair'  # 'pair' or 'cross' or 'normal'
    freeze_pretrained_untill = -1  # =0, 1, 2... <=0 when always freeze pretrained
    lr = 0.4
    gamma = 0.5
    warmup_till = -1  # <=1 when do not use warmup
    weight_decay = 5e-4
    momentum = 0.9
    margin = None
    sync_bn = False

    # evaluation optionsgit
    evaluate = False
    savefig = False
    re_ranking = False
    eval_step = 50
    test_batch = 2
    eval_phase_num = 1  # 1 / 2
    test_pids_num = -1  # = <0 when don't change test set
    eval_minors_num = 0  # <=0 when evaluation on the whole test set once
    eval_fast = False  # each query id has only one image for evaluation

    # model options
    model_name = 'braidmgn'  # braidnet, braidmgn, densebraidmgn, osnet
    feats = 256  # in braidmgn, pooled feature vector length in each part with one pooling method
    last_stride = 1
    pretrained_subparams = False
    pretrained_model = None
    disable_resume = False
    zero_tail_weight = False
    tail_times = 1
    
    # miscs
    print_freq = 30
    exp_name = 'test'
    exp_dir = './exps/test'

    # fig_dir = None

    def parse_(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt does not have attribute %s" % k)
            setattr(self, k, v)
        # print(self.dataset)
        self.exp_dir = join('./exps', self.exp_name)
        #self.fig_dir = join(self.exp_dir, 'visualize')

        if self.dataset[0] == '[':
            self.dataset = eval(self.dataset)
            # print(self.dataset)

        self.datatype = 'person'

        if isinstance(self.gpus, int):
            self.gpus = (self.gpus,)

        if self.model_name == 'braidmgn':
            self.pretrained_subparams = True

        if self.model_name == 'osnet':
            self.loss = 'lsce'
            self.train_mode = 'normal'
            self.train_phase_num = 1
            self.eval_phase_num = 2

    def state_dict_(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.endswith('_')}

    def print_(self):
        print('======== experiment config =========')
        pprint(self.state_dict_())
        print('=============== end ================')


opt = DefaultConfig()

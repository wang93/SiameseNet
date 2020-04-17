from __future__ import print_function, absolute_import

#import glob
import re
from os import path as osp
import os
import random
from abc import abstractmethod
random.seed(0)


"""Dataset classes"""


class __Dataset(object):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _check_before_run(self):
        pass

    @abstractmethod
    def _process_dir(self, dir_path, relabel=False):
        pass


class Market1501(__Dataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, dataset_dir, mode, root='/data/usersdata/wyc_datasets/Person'):
        self.dataset_dir = dataset_dir
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        train_relabel = (mode == 'retrieval')
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=train_relabel)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> {} loaded".format(dataset_dir))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_names = os.listdir(dir_path)
        img_paths = [os.path.join(dir_path, img_name) for img_name in img_names \
            if img_name.endswith('jpg') or img_name.endswith('png')]
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            #assert 0 <= pid <= 1501  # pid == 0 means background
            #assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


class MSMT17(__Dataset):
    """
    MSMT17
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    def __init__(self, dataset_dir, mode, root='/data/usersdata/wyc_datasets/Person'):
        self.dataset_dir = osp.join(root, dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        #self.test_dir = osp.join(self.dataset_dir, 'test')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train_relabel = (mode == 'retrieval')
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.list_train_path, relabel=train_relabel)
        val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, self.list_query_path, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, self.list_gallery_path, relabel=False)

        # train += val
        # num_train_imgs += num_val_imgs

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> MSMT17 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        #self.trainval = train
        #self.val = val
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        #self.num_trainval_pids = num_train_pids
        #self.num_val_pids = num_val_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        #self.images_dir = ''
        #self.num_cams = 15

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        # if not osp.exists(self.test_dir):
        #     raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path, relabel=False):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)
            camid = int(img_path.split('_')[2]) - 1
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        num_imgs = len(dataset)
        num_pids = len(pid_container)

        if relabel:
            dataset = [(img_path, pid2label[pid], camid) for img_path, pid, camid in dataset]

        return dataset, num_pids, num_imgs


def init_dataset(name, mode):
    root = '/data/usersdata/wyc-datasets/Person'
    if 'MSMT17' in name:
        return MSMT17(name, mode, root)
    return Market1501(name, mode, root)


def combine_train_sets(subsets):
    camid_maps = []
    camid_map_cur = 0
    for subset in subsets:
        cams = list(set([r[2] for r in subset]))
        map_ = {cam: camid_map_cur + i for i, cam in enumerate(cams)}
        camid_maps.append(map_)
        camid_map_cur += len(cams)

    pid_maps = []
    pid_map_cur = 0
    for subset in subsets:
        pids = list(set([r[1] for r in subset]))
        map_ = {pid: pid_map_cur + i for i, pid in enumerate(pids)}
        pid_maps.append(map_)
        pid_map_cur += len(pids)

    combined_set = []
    for subset, pid_map, camid_map in zip(subsets, pid_maps, camid_maps):
        for record in subset:
            record_new = (record[0], pid_map[record[1]], camid_map[record[2]])
            combined_set.append(record_new)

    num_pids = sum([len(m) for m in pid_maps])
    num_imgs = len(combined_set)

    return combined_set, num_pids, num_imgs


def __resampe_subsets(subsets):
    sample_nums = [len(s) for s in subsets]
    target_num = min(sample_nums)
    subsets = [random.sample(s, target_num) for s in subsets]
    return subsets


def combine_test_sets(query_sets, gallery_sets, resample=True):
    if resample:
        query_sets = __resampe_subsets(query_sets)
        gallery_sets = __resampe_subsets(gallery_sets)

    camid_maps = []
    camid_map_cur = 0
    pid_maps = []
    pid_map_cur = 0
    combined_query_set = []
    combined_gallery_set = []
    for query_set, gallery_set in zip(query_sets, gallery_sets):
        total_set = query_set + gallery_set

        cams = list(set([r[2] for r in total_set]))
        camid_map = {cam: camid_map_cur + i for i, cam in enumerate(cams)}
        camid_maps.append(camid_map)
        camid_map_cur += len(cams)

        pids = list(set([r[1] for r in total_set]))
        pid_map = {pid: pid_map_cur + i for i, pid in enumerate(pids)}
        pid_maps.append(pid_map)
        pid_map_cur += len(pids)

        for record in query_set:
            record_new = (record[0], pid_map[record[1]], camid_map[record[2]])
            combined_query_set.append(record_new)

        for record in gallery_set:
            record_new = (record[0], pid_map[record[1]], camid_map[record[2]])
            combined_gallery_set.append(record_new)

    num_query_pids = len(set([r[1] for r in combined_query_set]))
    num_gallery_pids = len(set([r[1] for r in combined_gallery_set]))
    num_query_imgs = len(combined_query_set)
    num_gallery_imgs = len(combined_gallery_set)

    return (combined_query_set, num_query_pids, num_query_imgs), (combined_gallery_set, num_gallery_pids, num_gallery_imgs)


def init_united_datasets(names, mode):
    datasets = [init_dataset(name, mode) for name in names]
    dataset_keep = datasets[0]
    train, num_train_pids, num_train_imgs = combine_train_sets([d.train for d in datasets])

    query_sets = [d.query for d in datasets]
    gallery_sets = [d.gallery for d in datasets]

    (query, num_query_pids, num_query_imgs),\
    (gallery, num_gallery_pids, num_gallery_imgs)\
        = combine_test_sets(query_sets, gallery_sets)

    del query_sets, gallery_sets, datasets

    dataset_keep.train = train
    dataset_keep.query = query
    dataset_keep.gallery = gallery
    dataset_keep.num_train_pids = num_train_pids
    dataset_keep.num_query_pids = num_query_pids
    dataset_keep.num_gallery_pids = num_gallery_pids

    num_total_pids = num_train_pids + num_query_pids
    num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

    print("=> United datasets of {} loaded".format(', '.join(names)))
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
    print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
    print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
    print("  ------------------------------")
    print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
    print("  ------------------------------")

    return dataset_keep



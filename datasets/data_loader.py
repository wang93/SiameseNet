from __future__ import print_function, absolute_import

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader, _DataLoaderIter, pin_memory_batch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


# class _PairLoaderIter(_DataLoaderIter):
#     def __init__(self, loader):
#         super(_PairLoaderIter, self).__init__(loader)
#
#     def __next__(self):
#         if self.num_workers == 0:  # same-process loading
#             indices = next(self.sample_iter)  # may raise StopIteration
#             batch = self.collate_fn([(self.dataset[i], self.dataset[j])for i, j in indices])
#             if self.pin_memory:
#                 batch = pin_memory_batch(batch)
#             return batch
#
#         # check if the next sample has already been generated
#         if self.rcvd_idx in self.reorder_dict:
#             batch = self.reorder_dict.pop(self.rcvd_idx)
#             return self._process_next_batch(batch)
#
#         if self.batches_outstanding == 0:
#             self._shutdown_workers()
#             raise StopIteration
#
#         while True:
#             assert (not self.shutdown and self.batches_outstanding > 0)
#             idx, batch = self._get_batch()
#             self.batches_outstanding -= 1
#             if idx != self.rcvd_idx:
#                 # store out-of-order samples
#                 self.reorder_dict[idx] = batch
#                 continue
#             return self._process_next_batch(batch)
#
#
# class PairLoader(DataLoader):
#     def __init__(self, *args, **kwargs):
#         super(PairLoader, self).__init__(*args, **kwargs)
#
#     def __iter__(self):
#         return _PairLoaderIter(self)


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

    def __len__(self):
        return len(self.dataset)


class ImagePairData(ImageData):
    def __getitem__(self, items):
        sample1 = ImageData.__getitem__(self, items[0])
        sample2 = ImageData.__getitem__(self, items[1])

        return sample1, sample2

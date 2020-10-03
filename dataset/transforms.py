# encoding: utf-8
import random

from PIL import Image
from torchvision import transforms as T

from dataset.random_erasing import RandomErasing, Cutout


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(
            round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


def pad_shorter(x):
    h,w = x.size[-2:]
    s = max(h, w) 
    new_im = Image.new("RGB", (s, s))
    new_im.paste(x, ((s-h)//2, (s-w)//2))
    return new_im


def bbox_worse(x, imsize,  p):
    if random.random() < p:
        expand_ratio = [random.uniform(1.,2.) for _ in range(2)]
        bimsize = [round(o*r) for o, r in zip(imsize, expand_ratio)]
        x = T.Resize(bimsize)(x)
        x = T.RandomCrop(imsize)(x)
    else:
        x = T.Resize(imsize)(x)
    return x


class TrainTransform(object):
    def __init__(self, data, meta, augmentaion=None):
        self.data = data
        self.imageSize = meta['imageSize']
        self.mean = meta['mean']
        self.std = meta['std']

        if augmentaion == 'Cutout':
            print('incorporate Cutout to augment training data')
            self.augment = Cutout(probability=0.5, size=self.imageSize[1]//2, mean=[0.0, 0.0, 0.0])
        elif augmentaion == 'RandomErasing':
            print('incorporate RandomErasing to augment training data')
            self.augment = RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        else:
            self.augment = lambda x: x

    def __call__(self, x):
        if self.data == 'person':
            x = T.Resize(self.imageSize)(x)
            #x = bbox_worse(x, (384, 128), 0.5)
        else:
            raise NotImplementedError

        x = T.RandomHorizontalFlip()(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=self.mean, std=self.std)(x)
        x = self.augment(x)
        return x

    def pre_process(self, x):
        if self.data == 'person':
            x = T.Resize(self.imageSize)(x)
            # x = bbox_worse(x, (384, 128), 0.5)
        else:
            raise NotImplementedError

        x = T.ToTensor()(x)
        x = T.Normalize(mean=self.mean, std=self.std)(x)

        return x

    def post_process(self, x):
        x = x.clone()  # for security
        x = T.RandomHorizontalFlip()(x)
        x = self.augment(x)
        return x


class TestTransform(object):
    def __init__(self, data, meta, flip=False):
        self.data = data
        self.flip = flip
        self.imageSize = meta['imageSize']
        self.mean = meta['mean']
        self.std = meta['std']

    def __call__(self, x=None):
        if self.data == 'person':
            x = T.Resize(self.imageSize)(x)
            #x = bbox_worse(x, (384, 128), 0.5)
        else:
            raise NotImplementedError

        if self.flip:
            x = T.functional.hflip(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=self.mean, std=self.std)(x)
        return x

    def pre_process(self, x):
        if self.data == 'person':
            x = T.Resize(self.imageSize)(x)
            # x = bbox_worse(x, (384, 128), 0.5)
        else:
            raise NotImplementedError

        if self.flip:
            x = T.functional.hflip(x)

        x = T.ToTensor()(x)
        x = T.Normalize(mean=self.mean, std=self.std)(x)

        return x

    def post_process(self, x):
        return x

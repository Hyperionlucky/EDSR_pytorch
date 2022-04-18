import cv2
import torch
import numpy as np
from skimage import transform, util
import random
from PIL import Image, ImageFilter, ImageEnhance
from utils import set_seed
set_seed.setup_seed(1997)


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, hr, lr, slope):
        for op in self.ops:
            hr, lr, slope = op(hr, lr, slope)
        return hr, lr, slope
# class Nomalinze(object):
#     def __init__(self,std=False):
#         self.im1_mean = [113.40864198,114.0898923,116.45767587]
#         self.im1_std = [48.3074962,46.27179584,48.14637872]
#         self.im2_mean = [111.15711222,114.22936278,118.250028522]
#         self.im2_std = [49.35331354,46.95115163,47.85695866]

#         self.std = std

#     def __call__(self, hr,lr,slope):
#         if not self.std:
#             T1 = T1/255.0
#             T2 = T2/255.0
#         else:
#             T1 =  (T1-self.im1_mean)/self.im1_std
#             T2 =  (T2-self.im2_mean)/self.im2_std

#         return hr,lr,slope


class Totensor(object):
    def __call__(self, hr, lr, slope):

        hr = hr[:, :, np.newaxis].astype(np.float32)
        hr = torch.from_numpy(hr)
        hr = hr.permute(2, 0, 1)

        lr = lr[:, :, np.newaxis].astype(np.float32)
        lr = torch.from_numpy(lr)
        lr = lr.permute(2, 0, 1)

        slope = slope[:, :, np.newaxis].astype(np.float32)
        slope = torch.from_numpy(slope)
        slope = slope.permute(2, 0, 1)

        return hr, lr, slope


class RandomScaleCrop(object):
    '''
    scale = [1,1,1,1,1.5,1.5,2,2.5]
    '''

    def __init__(self, crop_size, fill=0, scale=2):
        self.crop_size_h = crop_size
        self.crop_size_l = crop_size // scale
        self.fill = 0
        self.scale = scale

    def __call__(self, hr, lr, slope):

        H, W = lr.shape
        if H == self.crop_size_l and W == self.crop_size_l:
            return hr, lr, slope
        crop_h = random.randint(0, H-self.crop_size_l)
        crop_W = random.randint(0, W-self.crop_size_l)

        hr = hr[crop_h*self.scale:crop_h*self.scale+self.crop_size_h,
                crop_W*self.scale:crop_W*self.scale+self.crop_size_h]
        lr = lr[crop_h:crop_h+self.crop_size_l, crop_W:crop_W+self.crop_size_l]
        slope = slope[crop_h:crop_h+self.crop_size_l,
                      crop_W:crop_W+self.crop_size_l]

        return hr, lr, slope


class RandomHorizontalFlip(object):
    def __call__(self, hr, lr, slope):
        if random.random() < 0.5:
            hr = hr[:, ::-1].copy()
            lr = lr[:, ::-1].copy()
            slope = slope[:, ::-1].copy()

        return hr, lr, slope


class RandomVerticalFlip(object):
    def __call__(self, hr, lr, slope):
        if random.random() < 0.5:
            hr = hr[::-1, :].copy()
            lr = lr[::-1, :].copy()
            slope = slope[::-1, :].copy()
        return hr, lr, slope


class RandomRotation(object):
    def __init__(self, rotation_lists=[0, 90, 180, 270]):
        self.rotation_lists = rotation_lists

    def __call__(self, hr, lr, slope):
        self.rotation = random.choice(self.rotation_lists)
        hr = (transform.rotate(hr / 2000, self.rotation) * 2000).astype(np.float32)
        lr = (transform.rotate(lr / 2000, self.rotation) * 2000).astype(np.float32)
        slope = (transform.rotate(slope, self.rotation))

        return hr, lr, slope

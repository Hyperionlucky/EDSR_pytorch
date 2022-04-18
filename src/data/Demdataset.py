import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data.utils import transform
import random


class Demdataset(Dataset):
    def __init__(self, datapath, mode="train", crop_size = 96, scale=2, reverse=False):
        super(Demdataset, self).__init__()
        self.mode = mode
        with open(datapath, "r", errors='ignore') as lines:
            self.samples = []
            for line in lines:
                hr_path = line.strip().split(" ")[0]
                lr_path = line.strip().split(" ")[1]
                slope_path = line.strip().split(" ")[2]
                assert hr_path.split("_")[-1].split(".")[0] == lr_path.split("_")[-1].split(".")[0] == slope_path.split("_")[-1].split(".")[0]
                self.samples.append([hr_path, lr_path, slope_path])
        self.samples.sort()
        if mode == "train":
            self.transform = transform.Compose(transform.RandomHorizontalFlip(),
                                               transform.RandomVerticalFlip(),
                                               transform.RandomRotation(),
                                               transform.Totensor())
            self.reverse = reverse

            self.transform_scale = transform.Compose(transform.RandomScaleCrop(crop_size=crop_size, scale=scale),
                                                     transform.RandomHorizontalFlip(),
                                                     transform.RandomVerticalFlip(),
                                                    #  transform.RandomRotation(),
                                                     transform.Totensor())

        if mode == "val":
            self.transform = transform.Compose(transform.Totensor())
            self.reverse = False

    def __getitem__(self, index):
        hr_path, lr_path, slope_path = self.samples[index]
        if self.reverse:
            if random.random() < 0:
                hr = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
                lr = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
                slope = cv2.imread(slope_path, cv2.IMREAD_UNCHANGED)
                # label1 = cv2.imread(label2_path,cv2.IMREAD_UNCHANGED)
            else:
                hr = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
                lr = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
                slope = cv2.imread(slope_path, cv2.IMREAD_UNCHANGED)
        else:
            hr = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
            lr = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
            slope = cv2.imread(slope_path, cv2.IMREAD_UNCHANGED)

        if self.mode == "train":
            hr, lr, slope = self.transform_scale(hr, lr, slope)
        if self.mode == "val":
            hr, lr, slope = self.transform(hr, lr, slope)

        return hr, lr, slope

    def __len__(self):
        return len(self.samples)

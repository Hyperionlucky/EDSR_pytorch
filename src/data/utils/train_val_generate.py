# from functools import cache
import numpy as np
import random
import os

random.seed(27)
base_dir = "/home/cgd/DEM/ESDR_Pytorch/EDSR_pytorch"
hr_dir = base_dir + '/Dataset/HR'
lr_dir = base_dir + '/Dataset/LR/x2/Dem'
scope_dir = base_dir + '/Dataset/LR/x2/Slope'
SCALEOFTRAINVAL = 0.9

name_lists = os.listdir(hr_dir)

train_val_lists = []

for name in name_lists:
    if not name.endswith(".TIF"):
        continue
    cache = ""
    hr_path = os.path.join(os.path.abspath(hr_dir), name)
    lr_path = os.path.join(os.path.abspath(lr_dir), name)
    lr_path = lr_path.replace("192","96")
    scope_name = os.path.basename(lr_path).replace("Dem","Slope")
    scope_path = os.path.join(os.path.abspath(scope_dir), scope_name)
    cache = hr_path + " " + lr_path + " " + scope_path
    train_val_lists.append(cache)
print("num of datasets:", len(train_val_lists))
num_of_trainset = int(SCALEOFTRAINVAL * len(train_val_lists))
train_lists = random.sample(train_val_lists, num_of_trainset)
val_lists = []

for name in train_val_lists:
    if name not in train_lists:
        val_lists.append(name)
print("num of trainsets:", len(train_lists))
print("num of valsets:", len(val_lists))
with open(base_dir + "/Dataset/train_2x_slope.txt","w") as f:
    for name in train_lists:
        f.write(name+"\n")

with open(base_dir + "/Dataset/val_2x_slope.txt","w") as f:
    for name in val_lists:
        f.write(name+"\n")        
# from functools import cache
# import numpy as np
import random
import os

random.seed(27)
scale = [4]
SCALEOFTRAINVAL = 0.9

base_dir = "/home/ldx/DEM/ESDR_Pytorch/EDSR_pytorch"

for scale in scale:
    hr_dir = base_dir + '/Dataset_USA/HR/x' + str(scale)
    name_lists = os.listdir(hr_dir)
    lr_dir = base_dir + '/Dataset_USA/LR/x'+ str(scale) +'/Dem'
    flow_dir = base_dir + '/Dataset_USA/LR/x' + str(scale)  +'/Flow'
    train_val_lists = []

    for name in name_lists:
        if not name.endswith(".TIF"):
            continue
        cache = ""
        hr_path = os.path.join(os.path.abspath(hr_dir), name)
        lr_path = os.path.join(os.path.abspath(lr_dir), name)
        lr_path = lr_path.replace("192_192",str(192//scale)+"_"+str(192//scale))
        factor_name = os.path.basename(lr_path).replace("Dem","Flow")
        factor_name = factor_name.replace(str(192//scale)+"_"+str(192//scale),"192_192")
        factor_name = os.path.join(os.path.abspath(flow_dir), factor_name)
        cache = hr_path + " " + lr_path + " " + factor_name + "\n"
        train_val_lists.append(cache)
    print("num of datasets:", len(train_val_lists))
    num_of_trainset = int(SCALEOFTRAINVAL * len(train_val_lists)) if len(train_val_lists) < 5000 else 4500
    train_lists = random.sample(train_val_lists, num_of_trainset)
    # train_val_lists = random.sample(train_val_lists,len(train_val_lists))

    # with open(base_dir + "/Dataset_USA/train_" + str(scale)  +"x_flow.txt","r") as f_1:
        # array_train = f_1.readlines()
    # with open(base_dir + "/Dataset_USA/val_" + str(scale)  +"x_flow.txt","r") as f_2:
        # array_val = f_2.readlines()

    val_lists = []
    i = 0
    for name in train_val_lists:
        if name not in train_lists:
            if i == 500:
                break
            val_lists.append(name)
            i = i + 1
    print("num of trainsets:", len(train_val_lists))
    print("num of valsets:", len(val_lists))

    
    with open(base_dir + "/Dataset_USA/train_" + str(scale)  +"x_flow.txt","w") as f:
        for name in train_lists:
            f.write(name)

    with open(base_dir + "/Dataset_USA/val_" + str(scale)  +"x_flow.txt","w") as f:
        for name in val_lists:
            f.write(name)        
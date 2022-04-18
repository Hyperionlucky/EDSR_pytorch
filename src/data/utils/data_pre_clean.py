import numpy as np
import random
import os

random.seed(27)
base_dir = "/home/cgd/DEM/ESDR_Pytorch/EDSR_pytorch"
# hr_dir = base_dir + '/Dataset/HR'
lr_dir = base_dir + '/Dataset/LR/x4/Dem'
scope_dir = base_dir + '/Dataset/LR/x4/Slope'

name_lists = os.listdir(lr_dir)

for name in name_lists:
    fname = name.split('.')[0]
    if fname.split('_')[1] == 'patch2':
        if -1 <  int(fname.split('_')[-1]) < 65:
            cache = ""
            # hr_path = os.path.join(os.path.abspath(lr_dir), name)
            lr_path = os.path.join(os.path.abspath(lr_dir), name)
            # lr_path = lr_path.replace("192","96")
            slope_name = os.path.basename(lr_path).replace("Dem","Slope")
            slope_path = os.path.join(os.path.abspath(scope_dir), slope_name)
            cache = lr_path + " " + slope_path
            # os.remove(hr_path)
            os.remove(lr_path)
            os.remove(slope_path)
            print(cache)
            # train_val_lists.append(cache)

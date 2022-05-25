import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
from option import args

import model
from model.rfan import RFAN
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def toTensor(hr_path, lr_path):

    hr = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
    lr = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)

    hr = hr[:,:,np.newaxis].astype(np.float32)
    hr = torch.from_numpy(hr)
    hr = hr.permute(2, 0, 1).unsqueeze(dim=0)

    lr = lr[:,:,np.newaxis].astype(np.float32)
    lr = torch.from_numpy(lr)
    lr = lr.permute(2, 0, 1).unsqueeze(dim=0)
    return hr.cuda(), lr.cuda()

model_name = args.model
dataset_name = "Dataset_USA"
scale = args.scale
datasat_path = './' + dataset_name + '/val_' + str(scale) + 'x_flow.txt'
img_pred_dir = 'predict/{}_{}/x{}'.format(dataset_name, model_name, str(scale))

if not os.path.exists(img_pred_dir):
    os.makedirs(img_pred_dir)

samples = []

if datasat_path is not  None:
    with open(datasat_path, "r") as f:
        for line in f:
            samples.append([line.strip().split(" ")[0],
        line.strip().split(" ")[1]])

model_current =  RFAN(args)
model_current.cuda()
checkpoint = torch.load(args.resume, map_location='cpu')
model_current.load_state_dict(checkpoint['state_dict'])
model_current.eval()

for hr_path,lr_path in samples:
    img_name = hr_path.split(".")[0].split("/")[-1]
    hr,lr = toTensor(hr_path, lr_path)
    sr = model_current(lr)
    sr = sr.data.cpu().numpy()
    hr = hr.data.cpu().numpy()

    sr = np.squeeze(sr)
    hr = np.squeeze(hr)
    diff = np.abs(sr-hr)
    higer_q = np.quantile(diff, 0.8, interpolation='higher')
    # diff = 
    # mae = np.mean(diff)
    diff[diff < higer_q] = 0
    diff[diff > higer_q] = 1
    # diff = np.exp(diff)
    ax = sns.heatmap(diff, cmap="Greens", robust=True, xticklabels =False, yticklabels=False)
    figure = ax.get_figure()
    figure.savefig(os.path.join(img_pred_dir,img_name+".jpg"), dpi=1200)
    plt.clf()
    # mae = 







import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
from option import args
from osgeo import gdal
import model
import seaborn as sns

from utils.metrics import Evaluator

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
dataset_path = './' + dataset_name + '/val_' + str(scale) + 'x_terrain.txt'
img_pred_dir = 'predict/x{}/{}/{}/TestArea_SR'.format(str(scale),dataset_name, model_name )

if not os.path.exists(img_pred_dir):
    os.makedirs(img_pred_dir)

# 直接从文件夹中读取测试区域
hr_dir = dataset_name + "/TestArea/HR/x" + str(scale)
lr_dir = dataset_name + "/TestArea/LR/x" + str(scale) + '/Dem'

samples = []

name_lists = os.listdir(hr_dir)
for name in name_lists:
    if not name.endswith(".TIF"):
        continue
    hr_path = os.path.join(os.path.abspath(hr_dir), name)
    lr_path = os.path.join(os.path.abspath(lr_dir), name)
    lr_path = lr_path.replace("192_192",str(192//2)+"_"+str(192//2))
    samples.append([hr_path,lr_path])



# if dataset_path is not  None:
#     with open(dataset_path, "r") as f:
#         for line in f:
#             samples.append([line.strip().split(" ")[0],
#         line.strip().split(" ")[1]])
val_evaluator = Evaluator(batch_size = 1, scale=args.scale)
model_current =  model.Model(args=args)
model_current.cuda()
checkpoint = torch.load(args.resume, map_location='cpu')
model_current.load_state_dict(checkpoint['state_dict'])
model_current.eval()

mae_global = 0.0

for hr_path,lr_path in samples:
    img_name = hr_path.split(".")[0].split("/")[-1]
    hr,lr = toTensor(hr_path, lr_path)
    sr = model_current(lr)
    sr = sr.data.cpu().numpy()
    hr = hr.data.cpu().numpy()
    diff = np.squeeze(hr - sr)
    mae = np.mean(np.abs(diff))
    mae_global += mae
    sr = np.squeeze(sr)
    # 通过GDAL保存数据
    # 初始化
    img_proj = gdal.Open(hr_path).GetProjection()
    img_transf = gdal.Open(hr_path).GetGeoTransform()
    driver = gdal.GetDriverByName("GTiff")
    sr_tif = driver.Create(os.path.join(img_pred_dir,img_name+".tif"),192,192,1,gdal.GDT_Float32)
    sr_tif.SetGeoTransform(img_transf)
    sr_tif.SetProjection(img_proj)
    # 写入数据
    sr_tif.GetRasterBand(1).WriteArray(sr)
    # 2.163269
    # sr = np.squeeze(sr)
    # hr = np.squeeze(hr)
    # diff = np.abs(sr-hr)
    # higher_q = np.quantile(diff, 0.8, interpolation="higher")
    # # diff = 
    # # mae = np.mean(diff)
    # diff[diff < higher_q] = 0
    # diff[diff > higher_q] = 1
    # diff = np.exp(diff)
    # ax = sns.heatmap(diff, cmap="Greens", robust=True, xticklabels =False, yticklabels=False,vmax=2,vmin=0)
    # figure = ax.get_figure()
    # figure.savefig(os.path.join(img_pred_dir,"after.tif"), dpi=300)
    # plt.clf()
    # mae =
print(mae_global/len(samples)) 







import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class SlopeLossFunc(nn.Module):
    def __init__(self):
        super(SlopeLossFunc, self).__init__()

    def forward(self, sr, hr):
        sr = sr.cuda().data.cpu().numpy()
        sr.reshape(16, 96, 96)
        sr_slope = np.zeros((16, 96, 96))
        for i in range(16):
            j = sr[i, :, :]
            j_dx, j_dy = Cacdxdy(j)
            j_slope = CacSlopAsp(j_dx, j_dy)
            sr_slope[i, :, :] = j_slope
        hr = hr.cuda().data.cpu().numpy()
        hr.reshape(16, 96, 96)
        hr_slope = np.zeros((16, 96, 96))
        for i in range(16):
            j = hr[i, :, :]
            j_dx, j_dy = Cacdxdy(j)
            j_slope = CacSlopAsp(j_dx, j_dy)
            hr_slope[i, :, :] = j_slope
        result = abs(sr_slope - hr_slope)
        result.reshape(16, 1, 96, 96)
        loss = torch.mean(torch.tensor(result))

        return loss


def AddRound(npgrid):
    ny, nx = npgrid.shape[1:3]
    zbc = np.zeros((ny + 2, nx + 2))
    zbc[1:-1, 1:-1] = npgrid
    #
    # 四边
    zbc[0, 1:-1] = npgrid[0, 0, :]
    zbc[-1, 1:-1] = npgrid[0, -1, :]
    zbc[1:-1, 0] = npgrid[0, :, 0]
    zbc[1:-1, -1] = npgrid[0, :, -1]
    #角点
    zbc[0, 0] = npgrid[0, 0, 0]
    zbc[0, -1] = npgrid[0, 0, -1]
    zbc[-1, 0] = npgrid[0, -1, 0]
    zbc[-1, -1] = npgrid[0, -1, -1]

    return zbc


def Cacdxdy(npgrid, sizex=12.5, sizey=12.5):
    zbc = AddRound(npgrid)
    dx = (zbc[1:-1, :-2] - zbc[1:-1, 2:]) / (sizex * 2)
    dy = (zbc[2:, 1:-1] - zbc[:-2, 1:-1]) / (sizey * 2)
    return dx, dy


def CacSlopAsp(dx, dy):
    slope = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578
    return slope

import torch
import torch.nn as nn
# import torch.nn.functional as func
# import numpy as np


class SlopeLossFunc(nn.Module):
    def __init__(self, epsilon):
        super(SlopeLossFunc, self).__init__()
        self.eps = epsilon

    def forward(self, sr, hr):
        sr_offset_x = sr[:, :, :, 2:]
        hr_offset_x = hr[:, :, :, 2:]
        sr_offset_y = sr[:, :, 2:, :]
        hr_offset_y = hr[:, :, 2:, :]
        hr_diff_x = (hr[:, :, :, :-2] - hr_offset_x)[:,:,:-2,:] / 25
        sr_diff_x = (sr[:, :, :, :-2] - sr_offset_x)[:,:,:-2,:] / 25
        hr_diff_y = (hr[:, :, :-2, :] - hr_offset_y)[:,:,:,:-2] / 25
        sr_diff_y = (sr[:, :, :-2, :] - sr_offset_y)[:,:,:,:-2] / 25
        assert hr_diff_x.size() == hr_diff_y.size()
        assert sr_diff_x.size() == sr_diff_y.size()
        hr_slope = self.__cacSlope(hr_diff_x, hr_diff_y, self.eps)
        sr_slope = self.__cacSlope(sr_diff_x, sr_diff_y, self.eps)
        # calculate slope MAE
        loss = torch.mean(torch.abs(hr_slope - sr_slope))
        return loss


    def __cacSlope(self, dx, dy, eps):
        slope = (torch.arctan(torch.sqrt(dx ** 2 + dy ** 2 + eps))) * 57.29578
        return slope

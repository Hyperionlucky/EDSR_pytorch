import torch
import torch.nn as nn

class Loss(object):
    def __init__(self, weight=None):
        self.weight = weight
    def L1Loss(self, sr, hr):
        criterion = nn.L1Loss()
        loss = criterion(sr,hr)
        return loss

    def SlopeLoss(self, sr, hr):
        B,_,H,W = sr.size()
        sr.view(B, H, W)
        sr_slope = torch.zeros((B, H, W))
        for i in range(B):
            j = sr[i, :, :]
            j_dx, j_dy = self.Cacdxdy(j)
            j_slope = self.CacSlopAsp(j_dx, j_dy)
            sr_slope[i, :, :] = j_slope
        # hr = hr.cuda().data.cpu().numpy()
        hr.view(B, H, W)
        hr_slope = torch.zeros((B, H, W))
        for i in range(B):
            j = hr[i, :, :]
            j_dx, j_dy = self.Cacdxdy(j)
            j_slope = self.CacSlopAsp(j_dx, j_dy)
            hr_slope[i, :, :] = j_slope
        result = torch.abs(sr_slope - hr_slope)
        result.view(B, 1, H, W)
        loss = torch.mean(result)

        return loss
    def AddRound(self,npgrid):
        ny, nx = npgrid.size()[1:3]
        zbc = torch.zeros((ny + 2, nx + 2))
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


    def Cacdxdy(self,npgrid, size_x=12.5, size_y=12.5):
        zbc = self.AddRound(npgrid)
        dx = (zbc[1:-1, :-2] - zbc[1:-1, 2:]) / (size_x * 2)
        dy = (zbc[2:, 1:-1] - zbc[:-2, 1:-1]) / (size_y * 2)
        return dx, dy


    def CacSlopAsp(self,dx, dy):
        slope = (torch.arctan(torch.sqrt(dx * dx + dy * dy))) * 57.29578
        return slope

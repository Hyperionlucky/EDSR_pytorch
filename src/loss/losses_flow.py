import torch.nn as nn
import torch
from loss.slope_loss import SlopeLossFunc


class Loss(object):
    def __init__(self, weight=None):
        super(Loss, self).__init__()
        self.weight = weight
        self.slope_loss = SlopeLossFunc(epsilon=1e-8)

    def criterion(self, sr, hr):
        # loss1 = self.L1Loss(sr*flow, hr*flow)
        # flow_reverse = torch.abs(flow - 1)
        # loss2 = self.L1Loss(sr*flow_reverse, hr*flow_reverse)
        # return loss1 * self.weight[0] + loss2*self.weight[1]

        sr = sr.float()
        hr = hr.float()
        criterion = nn.L1Loss()
        return criterion(sr, hr)

    def terrain_criterion(self, sr, hr, terrain, epoch):
        loss = self.L1Loss(sr,hr)
        slope_loss = self.slope_loss(sr,hr)
        if epoch < 300 :
            return loss + self.weight[0] * slope_loss
        else: 
            terrain = terrain.data.cpu().numpy()
            terrain[terrain == 1] = 0
            terrain[terrain == 2] = 1
            terrain = torch.from_numpy(terrain).cuda()
            # loss1 = self.L1Loss(sr*terrain, hr*terrain)
            loss1 = self.terrain_loss(sr-hr, terrain)
            # + self.slope_loss(sr*terrain, hr*terrain)
            return loss + self.weight[0] * slope_loss + self.weight[1] * loss1
        # return loss

    def L1Loss(self, sr, hr):
        criterion = nn.L1Loss()
        return criterion(sr, hr)

    def BCELoss(self, sr, flow):
        logit = torch.sigmoid(sr)
        logit = logit.view(-1)
        target = flow.view(-1)
        loss = -torch.mean(self.weight[0] * target * torch.log(logit + 1e-7)
                           + self.weight[1] * (1 - target) * torch.log(1 - logit + 1e-7))
        return loss

    def dice_loss(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        smooth = 1.0
        i_flat = prediction.view(-1)
        t_flat = target.view(-1)
        intersection = (i_flat * t_flat).sum()
        return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

    def CrossEntropyLoss(self, logit, target, weight=torch.tensor([1, 1, 1]).float().cuda()):
        target = torch.squeeze(target, dim=1)
        criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')

        loss = criterion(logit, target.long())

        return loss
        

    def terrain_loss(self, diff, terrain):
        terrain_num = torch.sum(terrain>0.5)
        diff_flow = torch.sum(torch.abs(diff * terrain))
        terrain_num = terrain_num if terrain_num != 0 else 1
        loss = diff_flow / terrain_num
        return loss
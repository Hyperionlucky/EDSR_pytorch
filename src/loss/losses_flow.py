import torch.nn as nn
import torch


class Loss(object):
    def __init__(self, weight=None):
        super(Loss, self).__init__()
        self.weight = weight
        
    def criterion(self, sr, hr):
        # loss1 = self.L1Loss(sr*flow, hr*flow)
        # flow_reverse = torch.abs(flow - 1)
        # loss2 = self.L1Loss(sr*flow_reverse, hr*flow_reverse)
        # return loss1 * self.weight[0] + loss2*self.weight[1]
       
        sr = sr.float()
        hr = hr.float()
        criterion = nn.L1Loss()
        return criterion(sr, hr)
    def L1Loss(self, sr, hr):
        
        return 

    def BCELoss(self, sr, flow):
        logit = torch.sigmoid(sr)
        logit = logit.view(-1)
        target = flow.view(-1)
        loss = -torch.mean(self.weight[0] * target * torch.log(logit + 1e-7)
                           + self.weight[1] * (1 - target) * torch.log(1 - logit + 1e-7))
        return loss
    def CrossEntropyLoss(self, logit, target, weight = torch.tensor([1, 1, 1]).float().cuda()):
        target = torch.squeeze(target, dim=1)
        criterion = nn.CrossEntropyLoss(weight=weight,reduction='mean')

        loss = criterion(logit, target.long())

        return loss
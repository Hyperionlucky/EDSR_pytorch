import torch.nn as nn
import torch
class Loss(object):
    def __init__(self, weight=None):
        super(Loss, self).__init__()
        self.weight = weight
    def criterion(self, sr, hr, flow):
        loss1 = self.L1Loss(sr*flow, hr*flow)
        flow_reverse = torch.abs(flow -1)
        loss2 = self.L1Loss(sr*flow_reverse,hr*flow_reverse)
        return loss1 * self.weight[0] + loss2*self.weight[1]
    def L1Loss(self, sr,hr):
        criterion = nn.L1Loss()
        return criterion(sr,hr)
import torch.nn as nn
from loss.slope_loss import SlopeLossFunc as SlopeLoss

class Loss(object):
    def __init__(self, weight=None):
        self.weight = weight
    def L1Loss(self, sr, hr):
        criterion = nn.L1Loss()
        loss = criterion(sr,hr)
        return loss

    def SlopeLoss(self, sr, hr):
        criterion = SlopeLoss(epsilon=1e-8)
        loss = criterion(sr, hr)
        return loss
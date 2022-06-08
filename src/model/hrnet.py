import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.hrnet import HRNet32

def make_model(args):
    return HRNet(args)
def conv3x3(in_channels, out_channels,  atrous_rate=1):
    return nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False)
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.head = nn.Sequential(conv3x3(in_channels, inter_channels),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels, 1, bias=True))

    def forward(self, x):
        return self.head(x)
class HRNet(nn.Module):
    def __init__(self, args):
        super(HRNet, self).__init__()
        self.semantic_backbone = HRNet32(
            pretrained_path=None, norm_layer=None)
        self.head = FCNHead(480, 1)
    def forward(self, img, hr):
        img_feature = self.semantic_backbone(img)
        out = self.head(img_feature)
        out = F.interpolate(out, scale_factor=4, mode="bicubic", align_corners=False)

        return out

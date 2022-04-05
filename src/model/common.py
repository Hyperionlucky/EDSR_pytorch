import math
from statistics import mode
from numpy import isin

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

def weight_init(module):
    for n,m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity = 'relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m , (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity = 'relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488,), rgb_std=(1.0,), sign=-1):  # RGB均值和标准差

        super(MeanShift, self).__init__(1, 1, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(1).view(1, 1, 1, 1) / std.view(1, 1, 1, 1)  # tensor维度变换
        self.bias.data = sign * 10 * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):  # 每个残差模块两个卷积层 一个Relu
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class MResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MResBlock, self).__init__()
        m = []
        n = []
        for i in range(2):  # 每个残差模块两个卷积层 一个Relu
            if i == 0:
                m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
                n.append(conv(n_feats, n_feats, kernel_size + 2, bias=bias))
            else:
                m.append(conv(n_feats, n_feats // 2, kernel_size, bias=bias))
                n.append(conv(n_feats, n_feats // 2, kernel_size + 2, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats/2))
                n.append(nn.BatchNorm2d(n_feats / 2))
            if i == 0:
                m.append(act)
                n.append(act)

        self.body_3 = nn.Sequential(*m)
        self.body_5 = nn.Sequential(*n)
        self.res_scale = res_scale

    def forward(self, x):
        res_3 = self.body_3(x).mul(self.res_scale)
        res_5 = self.body_5(x).mul(self.res_scale)
        res = torch.cat((res_3, res_5), 1)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

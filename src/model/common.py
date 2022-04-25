import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def conv_3d(inchannels, out_channels, kernel_size, padding, bias=True):
    return nn.Conv3d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1, dilation=1, bias=bias)


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488,), rgb_std=(1.0,), sign=-1):  # RGB均值和标准差

        super(MeanShift, self).__init__(1, 1, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(1).view(
            1, 1, 1, 1) / std.view(1, 1, 1, 1)  # tensor维度变换
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


class nResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, key_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(nResBlock, self).__init__()
        m = []
        m.append(conv(key_feats, n_feats, kernel_size, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(
            conv(n_feats, key_feats, 1, bias=True),
            act,
            conv(key_feats, key_feats, 3, bias=True),
            act,
            conv(key_feats, n_feats, 1, bias=True)
        )
        self.res_scale = res_scale
        self.act = act

    def forward(self, x):
        # res = self.body(x).mul(self.res_scale)
        input_x = x
        res = self.body(x)
        res += input_x

        return self.act(res)


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
        res = self.body(x)
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
                n.append(conv(n_feats, n_feats // 2,
                         kernel_size + 2, bias=bias))
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


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DownBlock(nn.Module):
    def __init__(self, opt, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = 0.2

        if nFeat is None:
            nFeat = opt.n_feats

        if in_channels is None:
            in_channels = opt.n_channels

        if out_channels is None:
            out_channels = opt.n_channels

        self.dual_block = nn.Sequential(
            nn.Conv2d(in_channels, nFeat, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=negval, inplace=True),
            nn.Conv2d(nFeat, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False)
        )

        # for _ in range(1, int(math.log2(scale))):
        #     dual_block.append(
        #         nn.Sequential(
        #             nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
        #             nn.LeakyReLU(negative_slope=negval, inplace=True)
        #         )
        #     )

        # dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        # self.dual_module = dual_block

    def forward(self, x):
        x = self.dual_block(x)
        return x


class ESA(nn.Module):
    def __init__(self,  conv, channel,  reduction=4) -> None:
        super(ESA, self).__init__()
        self.channel_reduction = nn.Sequential(nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                                               nn.ReLU(True))
        conv_group = []
        for _ in range(3):
            conv_group.append(conv(channel//reduction, channel//reduction, 3))
            conv_group.append(nn.ReLU(True))
        self.conv_sa = nn.Sequential(
            # ke = k + (k-1) * (d-1) d:dilation
            nn.Conv2d(channel//reduction, channel//reduction,
                      kernel_size=3, dilation=6, padding=6, bias=True),
            nn.ReLU(True),
            *conv_group,
        )
        self.conv_du = nn.Sequential(
            conv(channel//reduction, channel, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.channel_reduction(x)
        feature_conv = self.conv_sa(feature)
        feature_sa = self.conv_du(feature+feature_conv)
        return feature_sa*x


class ERes(nn.Module):
    def __init__(self, conv, n_features) -> None:
        super(ERes, self).__init__()
        self.body = nn.Sequential(
            conv(n_features, n_features, 3),
            nn.ReLU(True),
            conv(n_features, n_features, 3),
            ESA(conv, n_features)
        )

    def forward(self, x):
        res = self.body(x)
        return res, res+x


class RFA(nn.Module):
    def __init__(self, conv, n_features) -> None:
        super(RFA, self).__init__()
        self.n_resblocks = 4
        rfa = [ERes(conv=conv, n_features=n_features)
               for _ in range(self.n_resblocks)]
        self.RFA = nn.ModuleList(rfa)
        self.tail = conv(n_features * 4, n_features, 1)

    def forward(self, x):
        feature = x
        res_feature = []
        for index in range(self.n_resblocks):
            tmp, x = self.RFA[index](x)
            res_feature.append(tmp)
        res = self.tail(torch.cat(res_feature, dim=1))
        res += feature
        return res

import math
from unittest import result
import torch
from model import common


import torch.nn as nn


class DRN(nn.Module):
    def __init__(self, args) -> None:
        super(DRN, self).__init__()
        conv = common.default_conv

        n_blocks = args.n_blocks # 40
        n_features = args.n_feats # 20
        act = nn.ReLU(True)

        scale = args.scale
        self.scale_factor = int(math.log2(scale))
        self.upsample = nn.Upsample(
            scale_factor=scale, mode='bicubic', align_corners=False)

        # define head module
        m_head = [conv(args.n_channels, n_features, 3),
                  nn.ReLU(True)]  # 第一个卷积层

        self.down = [common.DownBlock(args, nFeat=n_features*(2**p), in_channels=n_features*(2**p),
                                      out_channels=n_features*pow(2, p+1)) for p in range(self.scale_factor)]
        self.down = nn.ModuleList(self.down)


        up_RCAB_blocks = [[common.RCAB(conv=conv, n_feat=n_features * (2**p), kernel_size=3, act=act) for _ in range(n_blocks)] for p in range(self.scale_factor, 1, -1)]
        up_RCAB_blocks.insert(0, [common.RCAB(conv=conv, n_feat=n_features * (2**self.scale_factor), kernel_size=3, act=act) for _ in range(n_blocks)])
        # define body module
        upsample_blocks = [[common.Upsampler(conv, 2, n_feats=n_features*pow(2, self.scale_factor),act=False), conv(n_features*pow(2, self.scale_factor),n_features*pow(2, self.scale_factor-1),kernel_size=1)]]
        for p in range(self.scale_factor - 1, 0 ,-1):
            upsample_blocks.append([common.Upsampler(conv, 2, 2*n_features* (2 **p), act=False),conv(2*n_features*(2**p), n_features*pow(2,p-1), kernel_size=1)])
        self.up_blocks = nn.ModuleList()
        for i in range(self.scale_factor):
            self.up_blocks.append(nn.Sequential(*up_RCAB_blocks[i],  *upsample_blocks[i]))
        

        # define tail module
        m_tail = [ conv(n_features*pow(2,self.scale_factor), args.n_channels, 3)]
        for p in range(self.scale_factor, 0, -1):
            m_tail.append(conv(n_features*(2**p), args.n_channels, 3))
        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        common.weight_init(self)

    def forward(self, lr):
        # x = self.sub_mean(x)
        x = self.upsample(lr)
        x = self.head(x)

        copies = []
        for i in range(self.scale_factor):
            copies.append(x)
            x = self.down[i](x)
        sr_lr = self.tail[0](x)
        results = [sr_lr]
        for i in range(self.scale_factor):
            x = self.up_blocks[i](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.scale_factor - i - 1]), 1)
            # output sr imgs
            sr = self.tail[i + 1](x)
            results.append(sr)
        # x = self.add_mean(x)

        return results

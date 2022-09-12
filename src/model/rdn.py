import torch
from model import common


import torch.nn as nn


def make_model(args):
    return RDN(args)

class RDN(nn.Module):
    def __init__(self, args) -> None:
        super(RDN, self).__init__()
        conv = common.default_conv

        self.G0 = 64
        self.G = 64
        self.D = 30
        self.C = 8
        act = nn.ReLU(True)

        scale = args.scale

        # define head module
        self.sfe1 = nn.Conv2d(1, 64, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2)
        # m_head = [conv(args.n_channels, n_features, 3)]  # 第一个卷积层
        # residual dense blocks
        self.rdbs = nn.ModuleList([common.RDB(self.G0, self.G, self.C)])

        for _ in range(self.D - 1):
            self.rdbs.append(common.RDB(self.G, self.G, self.C))

        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )


        # define tail module
        m_tail = [  # 上采样
            common.Upsampler(conv, scale, 64, act=False),
            conv(64, args.n_channels, 3)
        ]
        self.tail = nn.Sequential(*m_tail)
        common.weight_init(self)

    def forward(self, x):
        # x = self.sub_mean(x)
        # x = lr
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        # res += x

        x = self.tail(x)
        # x = self.add_mean(x)

        return x

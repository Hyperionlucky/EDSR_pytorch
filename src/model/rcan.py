from model import common


import torch.nn as nn


def make_model(args):
    return RCAN(args)


class RCAN(nn.Module):
    def __init__(self, args) -> None:
        super(RCAN, self).__init__()
        conv = common.default_conv

        # n_resblocks = args.n_resblocks
        n_features = 64
        act = nn.ReLU(True)

        scale = args.scale

        # define head module
        m_head = [conv(args.n_channels, n_features, 3)]  # 第一个卷积层

        # define body module
        m_body = [  # 残差块
            common.ResidualGroup(
                conv, 64, 3, 16, n_resblocks=20) \
            for _ in range(10)
        ]
        m_body.append(conv(n_features, n_features, 3))  # 卷积层

        # define tail module
        m_tail = [  # 上采样
            common.Upsampler(conv, scale, n_features, act=False),
            conv(n_features, args.n_channels, 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        common.weight_init(self)

    def forward(self, x):
        # x = self.sub_mean(x)
        # x = lr
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

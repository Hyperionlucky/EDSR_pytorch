import torch
from model import common


import torch.nn as nn

def make_model(args):
    return EDSR(args)
class EDSR(nn.Module):
    def __init__(self, args) -> None:
        super(EDSR, self).__init__()
        conv = common.default_conv
        
        n_resblocks = args.n_resblocks
        n_features = args.n_features
        act = nn.ReLU(True)

        scale = args.scale


        # define head module
        m_head = [conv(args.n_channels, n_features, 3)]        #第一个卷积层

        # define body module
        m_body = [                                                                #残差块
            common.ResBlock(
                conv, n_features, 3, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_features, n_features, 3))                #卷积层

        # define tail module
        m_tail = [                                                       #上采样
            common.Upsampler(conv, scale, n_features, act=False),
            conv(n_features, args.n_channels, 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        common.weight_init(self)
    def forward(self, lr):
        # x = self.sub_mean(x)
        # x = lr
        x = self.head(lr)
        
        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x



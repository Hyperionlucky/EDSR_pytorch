import torch
from model import common
import torch.nn.functional as F

import torch.nn as nn

def make_model(args):
    return SRCNN(args)
class SRCNN(nn.Module):
    def __init__(self, args) -> None:
        super(SRCNN, self).__init__()
        conv = common.default_conv
        
        # n_resblocks = args.n_resblocks
        n_features = args.n_features
        act = nn.ReLU(True)

        self.scale = args.scale


        # define head module
        m_head = [conv(args.n_channels, n_features, 3),act]        #第一个卷积层

        # define body module
        m_body = [                                                                #残差块
            conv(n_features, 32, 1),act
        ]
        # m_body.append(conv(n_features, n_features, 3))                #卷积层

        # define tail module
        m_tail = [                                                       #上采样
            # common.Upsampler(conv, scale, n_features, act=False),
            conv(32, args.n_channels, 5)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        common.weight_init(self)
    def forward(self, lr):
        # x = self.sub_mean(x)
        # x = lr
        lr = F.interpolate(lr.float(), scale_factor=self.scale, mode="bicubic")
        x = self.head(lr)
        
        res = self.body(x)
        # res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

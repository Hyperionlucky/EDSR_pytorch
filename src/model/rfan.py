import torch
from model import common

import torch.nn as nn

def make_model(args):
    return RFAN(args)
class RFAN(nn.Module):
    def __init__(self, args) -> None:
        super(RFAN, self).__init__()
        conv = common.default_conv
        
        # self.n_resblocks = 4
        self.n_rfanblocks = 30
        n_features = 64

        scale = args.scale

        act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_channels, n_features, 3)]        #第一个卷积层

        # define body module
        m_body = [                                                                #残差块
            common.RFA(conv, n_features, scale=scale)
         for _ in range(self.n_rfanblocks)]             #卷积层
        # m_body.append()    
        # define tail module
        m_tail = [
            conv(n_features, n_features, 3),
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
        
        # for i in range(self.)
        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x



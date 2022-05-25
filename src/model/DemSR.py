import torch
from model import common


import torch.nn as nn

def make_model(args):
    return DemSR(args)

class DemSR(nn.Module):
    def __init__(self, args):
        super(DemSR, self).__init__()
        conv = common.default_conv
        
        n_resblocks = args.n_resblocks
        n_features = args.n_features
        act = nn.ReLU(True)

        scale = args.scale


        # define head module
        m_head = [conv(args.n_channels*2, n_features, 3),nn.ReLU(True)]        #第一个卷积层

        # define body module
        m_body = [                                                                #残差块
            common.nResBlock(
                conv, n_features,64, 3, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_features, n_features, 3))                #卷积层

        # define tail module
        m_tail = [                                                       #上采样
            common.Upsampler(conv, scale, n_features, act="relu"),
            conv(n_features, args.n_channels, 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        common.weight_init(self)
    def forward(self, lr,slope):
        # x = self.sub_mean(x)
        x = torch.cat([lr,slope],dim=1)
        x = self.head(x)
        
        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x



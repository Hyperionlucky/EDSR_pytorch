from collections import OrderedDict
import torch
from model import common
from model.utils.attention import NonLocalAttention
import torch.nn as nn
from model.hrnet import HRNet


def make_model(args):
    return RFAN(args)


def load_model(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[13:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


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
        m_head = [conv(args.n_channels, n_features, 3)]  # 第一个卷积层
        # define body module
        # attention = nn.Sequential(NonLocalAttention(conv=conv, channel=n_features * 4, reduction=4),
        # conv(n_features * 4,n_features,1)
        #   )
        m_body = [ common.EResidualGroup(conv=conv, n_feat=n_features, n_resblocks=10) for _ in range(3)]
        # RFA_Block(conv=conv, n_features=n_features,
                        #    scale=scale, num_blocks=self.n_rfanblocks, act=act)  # 卷积层
        # m_body.append()
        # define tail module
        m_tail = [
            conv(n_features, n_features, 3),
            common.Upsampler(conv, scale, n_features, act=False),
            conv(n_features, args.n_channels, 3)
        ]

        # self.terrain_tail = nn.Sequential(
        #     conv(args.n_channels, n_features, 3),
        #     common.RFA(conv = conv, n_features=n_features,scale=scale),
        #     conv(n_features, args.n_channels, 3))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        # self.model_cls = HRNet(args=args)
        common.weight_init(self)
        # for child in self.model_cls.children():
        #     for param in child.parameters():
        #         param.requires_grad = False
        # if args.pretrained_path is not None:
        #     self.model_cls = load_model(self.model_cls, args.pretrained_path)

    def forward(self, lr):
        # x = self.sub_mean(x)
        # x = lr
        x = self.head(lr)
        res = self.body(x)

        res += x

        results = self.tail(res)
        # x = self.add_mean(x)
        # terrain_line = self.model_cls(results)
        # hr_line = self.model_cls(hr)
        return results


class RFA_Block(nn.Module):
    def __init__(self, conv, n_features, scale, num_blocks, act) -> None:
        super(RFA_Block, self).__init__()
        self.num_blocks = num_blocks
        m_body = [common.RFA(conv=conv, n_features=n_features, scale=scale)
                  for _ in range(num_blocks)]
        # attention_list = [NonLocalAttention(conv=conv, channel=n_features, reduction=4)
        #                                 for _ in range(3)]
        # for i in range(10,num_blocks+10,10):
        #     m_body.insert(i, attention_list[i//10 - 1])
        self.body = nn.ModuleList(m_body)
        self.tail = nn.Sequential(conv(n_features*3, n_features, 1))

    def forward(self, x):
        local_features = []
        for i in range(self.num_blocks):
            x = self.body[i](x)
            if (i+1) % 10 == 0:
                local_features.append(x)
        return self.tail(torch.cat(local_features, dim=1))


class RFA_Attention(nn.Module):
    def __init__(self, attention, n_features, scale, num_blocks, act) -> None:
        super(RFA_Attention, self).__init__()
        self.num_blocks = num_blocks
        self.body = nn.ModuleList([common.RFA_NoTail(
            conv=common.default_conv, n_features=n_features, scale=scale, act=act) for _ in range(num_blocks)])
        self.attention = attention

    def forward(self, x):
        for i in range(self.num_blocks):
            res = self.body[i](x)
            res = self.attention(res)
            x += res
        return x

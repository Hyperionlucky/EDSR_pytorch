import math
# from unittest import result
import torch
from model import common


import torch.nn as nn


class Dual(nn.Module):
    def __init__(self, args) -> None:
        super(Dual, self).__init__()
        # conv = common.default_conv

        # act = nn.ReLU(True)

        scale = args.scale
        # self.scale_factor = int(math.log2(scale))
        self.dual_model = common.DownBlock(args, scale)
        common.weight_init(self)

    def forward(self, x):
        # x = self.sub_mean(x)
        return self.dual_model(x)
        # x = self.add_mean(x)
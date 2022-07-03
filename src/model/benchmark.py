import torch.nn as nn
import torch.nn.functional as F

def make_model(args):
    return Benchmark(args)
class Benchmark(nn.Module):
    def __init__(self, args) -> None:
        super(Benchmark, self).__init__()
        self.scale = args.scale
        self.method = "bicubic"
    def forward(self, lr):
        # H,W = lr.size(2), lr.size(3)
        sr = F.interpolate(lr.float(), scale_factor=self.scale, mode=self.method)
        return sr
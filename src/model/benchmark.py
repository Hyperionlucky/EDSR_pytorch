import torch.nn as nn
import torch.nn.functional as F

class Benchmark(nn.Module):
    def __init__(self, args) -> None:
        super(Benchmark, self).__init__()
        self.scale = args.scale
        self.method = args.model
    def forward(self, lr, slope):
        # H,W = lr.size(2), lr.size(3)
        sr = F.interpolate(lr.float(), scale_factor=self.scale, mode=self.method)
        return sr
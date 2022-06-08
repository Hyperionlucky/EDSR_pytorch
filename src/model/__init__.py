import os
from importlib import import_module

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args) -> None:
        super(Model, self).__init__()
        print('making model……')
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
    def forward(self, x):
        return self.model(x)

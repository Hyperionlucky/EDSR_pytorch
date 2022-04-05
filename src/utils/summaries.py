import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summart(self):
        writer = SummaryWriter(logdir=self.directory)
        return writer

    def visualize_image(self, writer, hr, lr, slope, sr, global_step, mode):
        hr = hr.float()
        hr = hr[:4]
        grid_hr = make_grid(hr, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'hr'), grid_hr, global_step)

        lr = lr.float()
        lr = lr[:4]
        grid_lr = make_grid(lr, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'lr'), grid_lr, global_step)

        sr = sr.float()
        sr = sr[:4]
        grid_sr = make_grid(sr, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'sr'), grid_sr, global_step)

        slope = slope.float()
        slope = slope[:4]
        grid_slope = make_grid(slope, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, 'slope'), grid_slope, global_step)

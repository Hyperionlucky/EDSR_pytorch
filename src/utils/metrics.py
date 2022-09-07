import numpy as np
import math
import random
import torch

RGB_RANGE = 4294967296


class Evaluator(object):
    def __init__(self, batch_size=16, rgb_range=RGB_RANGE, scale = 3) -> None:
        self.batch_size = batch_size
        self.rgb_range = rgb_range
        self.scale = 20 if scale == 3 else 10
        self.metric_matrix = [np.zeros((batch_size, 1)) for _ in range(7)]

    def score(self, num):
        if num % self.batch_size != 0:
            num_remain = num % self.batch_size
            num -= num_remain
        result = [ np.sum(i) / num for i in self.metric_matrix]
        return result

    def __generate_matrix(self, hr, sr, flow):
        diff = hr -sr
        mse = np.mean(diff ** 2, axis=(2, 3))
        # if np.max(mse) > 100:
        #     i = 1
        psnr = -10 * np.log10(mse/(self.rgb_range**2))
        mae = np.mean((np.abs(diff)), axis=(2, 3))
        rmse = np.sqrt(mse)
        terrain_num = np.sum(flow == 1, axis = (2,3))
        terrain_num[terrain_num == 0 ] = 1
        flow_mae = np.sum((np.abs(diff * flow)), axis=(2,3)) / terrain_num
        e_max = np.max(np.abs(diff), axis=(2, 3))
        slope_mae = self.cac_slope_mae(sr,hr)
        return [mse, mae, rmse, e_max, flow_mae, psnr, slope_mae]

    def add_batch(self, sr, hr, flow):
        assert hr.shape == sr.shape
        # slope_mae = self.cac_slope_mae(sr, hr)
        # diff = hr - sr
        matrix = self.__generate_matrix(hr, sr, flow=flow)
        # matrix.insert(4, slope_mae)
        self.metric_matrix = [i + j for i,j in zip(self.metric_matrix, matrix)]

    def reset(self):
        self.metric_matrix = [np.zeros((self.batch_size, 1)) for _ in range(7)]

    def __cacSlope(self, dx, dy):
        slope = (np.arctan(np.sqrt(dx ** 2 + dy ** 2))) * 57.295779513
        return slope

    def cac_slope_mae(self, sr, hr):
        sr_offset_x = sr[:, :, :, 2:]
        hr_offset_x = hr[:, :, :, 2:]
        sr_offset_y = sr[:, :, 2:, :]
        hr_offset_y = hr[:, :, 2:, :]
        hr_diff_x = (hr[:, :, :, :-2] - hr_offset_x)[:, :, :-2, :] / self.scale
        sr_diff_x = (sr[:, :, :, :-2] - sr_offset_x)[:, :, :-2, :] / self.scale
        hr_diff_y = (hr[:, :, :-2, :] - hr_offset_y)[:, :, :, :-2] / self.scale
        sr_diff_y = (sr[:, :, :-2, :] - sr_offset_y)[:, :, :, :-2] / self.scale
        hr_slope = self.__cacSlope(hr_diff_x, hr_diff_y)
        sr_slope = self.__cacSlope(sr_diff_x, sr_diff_y)
        slope_mae = np.mean((np.abs(hr_slope - sr_slope)), axis=(2, 3))
        return slope_mae

    # def add_batch(self, hr, sr):
    # sr = sr.astype(np.uint8)
if __name__ == '__main__':
    mse, psnr, mae, rmse, e_max, slope_mae = [random.randint(0,9)  for _ in range(6)]
    metric_dict = {
        "MSE": mse,
        "PSNR": psnr,
        "MAR": mae,
    }
    with open("best_metrics.txt", "w") as f:
        f.write(str(metric_dict))
    # mse = mse + 1 + 2 + 3
    # hr =  torch.randn(16,1,192,192)
    c = 1

from matplotlib.pyplot import axis
import numpy as np
import math

import torch

RGB_RANGE = 65535


class Evaluator(object):
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.mse_matrix = np.zeros((batch_size, 1))
        self.psnr_matrix = np.zeros((batch_size, 1))
        self.mae_matrix = np.zeros((batch_size, 1))
        self.rmse_matrix = np.zeros((batch_size, 1))
        # self.sr = sr

    def score(self,num):
        # MAE = np.mean(np.abs(self.hr-self.sr))
        # return MAE
        # np.sum()
        mse = np.sum(self.mse_matrix,axis=0) / num
        psnr = np.sum(self.psnr_matrix,axis=0) / num
        mae = np.sum(self.mae_matrix,axis=0) / num
        rmse = np.sum(self.rmse_matrix,axis=0) / num
        return mse,psnr,mae,rmse

    def _generate_matrix(self, diff):
        mse = np.mean(diff ** 2,axis=(2, 3))
        # a = mse/torch.tensor(RGB_RANGE)
        # if mse < 1e
        # a = mse / (RGB_RANGE**2)
        # psnr = -10 * math.log10(a)
        psnr = -10 * np.log10(mse/(RGB_RANGE**2))
        mae = np.mean((np.abs(diff)),axis=(2, 3))
        rmse = np.sqrt(mse)
        return [mse, psnr, mae, rmse]

    def add_batch(self, sr, hr):
        assert hr.shape == sr.shape
        diff = (hr - sr)[:, :, 2:-2, 2:-2]
        matrix = self._generate_matrix(diff=diff)
        self.mse_matrix += matrix[0]
        self.psnr_matrix += matrix[1]
        self.mae_matrix += matrix[2]
        self.rmse_matrix += matrix[3]
    def reset(self):
        self.mse_matrix = np.zeros((self.batch_size, 1))
        self.psnr_matrix = np.zeros((self.batch_size, 1))
        self.mae_matrix = np.zeros((self.batch_size, 1))
        self.rmse_matrix = np.zeros((self.batch_size, 1))
    # def add_batch(self, hr, sr):
    # sr = sr.astype(np.uint8)
if __name__ == '__main__':
    hr = torch.tensor([[[[1,5,6],[15,30,90],[40,80,150]]]])
    sr = torch.randn(16, 1, 192, 192)
    diff = torch.diff(hr, n=2,dim=3)


    ev = Evaluator(16)
    for i in range(20):
        hr = torch.randn(16, 1, 192, 192)
        sr = torch.randn(16, 1, 192, 192)
        ev.add_batch(sr,hr)
    mse, psnr, mae, rmse = ev.score()

    # hr =  torch.randn(16,1,192,192)
    c = 1

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import re
import math

class OwnOp(Function):
    def forward(input_tensor):
        tensor = input_tensor.numpy()
        tensor = AddRound(tensor)
        dx, dy = Cacdxdy(tensor)
        slope = CacSlopAsp(dx, dy)
        result = slope
        return torch.tensor(result)

    def backward(grad_output):

        return grad_output

def AddRound(npgrid):
    ny, nx = npgrid.shape()
    zbc = np.zeros((ny+2,nx+2))
    zbc[1:-1,1:-1]=npgrid

    #四边
    zbc[0,1:-1]=npgrid[0,:]
    zbc[-1,1:-1]=npgrid[-1,:]
    zbc[1:-1,0]=npgrid[:,0]
    zbc[1:-1,-1]=npgrid[:,-1]
    #角点
    zbc[0,0]=npgrid[0,0]
    zbc[0,-1]=npgrid[0,-1]
    zbc[-1,0]=npgrid[-1,0]
    zbc[-1,-1]=npgrid[-1,-1]

def Cacdxdy(npgrid, sizex=12.5, sizey=12.5):
    zbc=AddRound(npgrid)
    dx=((zbc[1:-1,:-2]-zbc[1:-1,2:]))/sizex/2/1000
    dy=((zbc[2:,1:-1]-zbc[:-2,1:-1]))/sizey/2/1000
    dx=dx[1:-1,1:-1]
    dy=dy[1:-1,1:-1]
    return dx,dy

def CacSlopAsp(dx,dy):
    slope=(np.arctan(np.sqet(dx*dx+dy*dy)))*57.29578
    slope=slope[1:-1,1:-1]
    return slope;
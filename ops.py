# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

OPS = {
    'avg_pool_3x3': lambda C, stride, affine: AvgPool('avg', C, 3, stride, 1, affine=affine),
    'avg_pool_2x2': lambda C, stride, affine: AvgPool('avg', C, 2, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_2x2': lambda C, stride, affine: SepConv(C, C, 2, stride, 1, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  
    'dil_conv_2x2': lambda C, stride, affine: DilConv(C, C, 2, stride, 1, 2, affine=affine)
}

PRIMITIVES = [
    'avg_pool_3x3',
    'avg_pool_2x2',
    'skip_connect',  # identity
    'sep_conv_3x3',
    'sep_conv_2x2',
    'dil_conv_3x3',
    'dil_conv_2x2',
]


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PhotonicSigmoid()
        )
    
    def forward(self, x):
        return self.op(x)
        
                
        
class PhotonicSigmoid(nn.Module):
    def forward(self, x):
        tmp = torch.exp((x - 0.145) / 0.073)
        tmp = 1.005 + (0.06 - 1.005) / (1 + tmp)
        return tmp.float()


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            PhotonicSigmoid()
        )
    
    def forward(self, x):
        return self.op(x)

class AvgPool(nn.Module):
    """
    AvgPool with BN.
    """
    def __init__(self, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        self.bn = nn.BatchNorm2d(C, affine=affine)
        
    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out
        
        
class PhotonicSigmoid(nn.Module):
    def forward(self, x):
        tmp = torch.exp((x - 0.145) / 0.073)
        tmp = 1.005 + (0.06 - 1.005) / (1 + tmp)
        return tmp.float()

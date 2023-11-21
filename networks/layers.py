import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv3x3, self).__init__()

        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels, bias)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def subpixelconvolution(x):
    ps = nn.PixelShuffle(4)
    return ps(x)



class Concat(nn.Module):
    def __init__(self, channels, **kwargs):
        super(Concat, self).__init__()
        self.conv = nn.Conv2d(channels*2, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, equi_feat, c2e_feat):

        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.conv(x))
        return x



class add(nn.Module):
    def __init__(self, channels, SE=True):
        super(add, self).__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(channels * 2, channels, 1, bias=False)
    def forward(self, equi_feat, sph_feat):
        b, c, h, w = equi_feat.shape
        aa = h//3
        mid_equi_feat = equi_feat[:, :, aa:aa * 2, :]
        w0 = torch.exp(self.w[0] / torch.sum(torch.exp(self.w)))
        w1 = torch.exp(self.w[1] / torch.sum(torch.exp(self.w)))
        fuse_td = w0*equi_feat + w1*sph_feat
        fuse_td = torch.cat([equi_feat, fuse_td], 1)
        fuse_td = self.conv(fuse_td)
        fuse_td[:, :, aa:aa * 2, :] = mid_equi_feat
        fuse_fea = self.relu(fuse_td)

        return fuse_fea
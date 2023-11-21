from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from .convnext import *
import copy
from .blocks import Transformer_Block
from .ViT import miniViT, layers

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, subpixelconvolution, Concat, add

from collections import OrderedDict


class sconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.sphere_conv = nn.Sequential(
            nn.Conv2d(in_channels*9, in_channels, kernel_size=1, stride=stride,padding=0, groups=in_channels),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(False)
        )
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape

        return torch.nn.functional.grid_sample(inp, deformation, align_corners=True)
    def forward(self, x, lut):
        B, C, H, W = x.shape
        x_mid = self.deform_input(x, lut[0])
        x_left = self.deform_input(x, lut[1])
        x_left_up = self.deform_input(x, lut[2])
        x_up = self.deform_input(x, lut[3])
        x_right_up = self.deform_input(x, lut[4])
        x_right = self.deform_input(x, lut[5])
        x_right_down = self.deform_input(x, lut[6])
        x_down = self.deform_input(x, lut[7])
        x_left_down = self.deform_input(x, lut[8])
        catx = torch.stack((x_mid, x_left, x_left_up, x_up, x_right_up, x_right, x_right_down, x_down, x_left_down),dim=2).reshape(B, C * 9, H, W)
        return self.sphere_conv(catx)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.sconv1 = sconv(in_channels, mid_channels, 2)
        self.sconv2 = sconv(mid_channels, out_channels, 1)
    def forward(self, x,lut,lut1):
        x1 = self.sconv1(x,lut)
        x1 = self.sconv2(x1,lut1)
        return x1
class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.sconv1 = sconv(in_channels, mid_channels, 4)
        self.sconv2 = sconv(mid_channels, out_channels, 1)
    def forward(self, x,lut,lut1):
        x1 = self.sconv1(x,lut)
        x1 = self.sconv2(x1,lut1)
        return x1

class DoubleConv3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.sconv1 = sconv(in_channels, mid_channels, 1)
        self.sconv2 = sconv(mid_channels, out_channels, 1)
    def forward(self, x,lut):
        x1 = self.sconv1(x,lut)
        x1 = self.sconv2(x1,lut)
        return x1

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x, lut,lut1):
        return self.conv(x,lut,lut1)

class SphFuse512(nn.Module):

    def __init__(self, num_layers, equi_h, equi_w, pretrained=False, max_depth=10.0,
                 fusion_type="cee", se_in_fusion=True,nbins=100, min_val=0.1, max_val=10):
        super(SphFuse512, self).__init__()
        self.num_classes = nbins
        self.min_val = min_val
        self.max_val = max_val
        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion

        # encoder
        self.equi_encoder = convnext_base(pretrained)
        self.inc = DoubleConv2(3, 128)
        self.conv = DoubleConv3(128, 128)
        self.down1 = Down(128, 256)  # 128
        self.down2 = Down(256, 512)  # 64
        self.down3 = Down(512, 1024)  # 32

        self.num_ch_enc = np.array([128, 128, 256, 512, 1024])  #
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        self.equi_dec_convs = OrderedDict()
        # self.c2e = {}

        Fusion_dict = {"cat": Concat,
                       "cee": add}
        FusionLayer = Fusion_dict[self.fusion_type]



        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0]*16)

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))

        self.sigmoid = nn.Sigmoid()

        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)


    def forward(self, input_equi_image,lut512,lut128,lut64,lut32):

        equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4 = self.equi_encoder(input_equi_image)
        sph_enc_feat0 = self.inc(input_equi_image, lut512, lut128)
        sph_enc_feat1 = self.conv(sph_enc_feat0, lut128)
        sph_enc_feat2 = self.down1(sph_enc_feat1, lut128, lut64)
        sph_enc_feat3 = self.down2(sph_enc_feat2, lut64, lut32)

        outputs = {}

        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_enc_feat4))


        fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, sph_enc_feat3)
        equi_x = torch.cat([equi_x, fused_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))


        fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, sph_enc_feat2)
        equi_x = torch.cat([equi_x, fused_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))


        fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, sph_enc_feat1)
        equi_x = torch.cat([equi_x, fused_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)
        equi_x = self.equi_dec_convs["upconv_2"](equi_x)


        fused_feat0 = self.equi_dec_convs["fusion_1"](equi_enc_feat0, sph_enc_feat0)
        equi_x = torch.cat([equi_x, fused_feat0], 1)
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = subpixelconvolution(self.equi_dec_convs["upconv_1"](equi_x))

        equi_x = self.equi_dec_convs["deconv_0"](equi_x)


        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)
        outputs["equi_enc_feat0"] = equi_enc_feat0
        outputs["equi_enc_feat1"] = equi_enc_feat1
        outputs["equi_enc_feat2"] = equi_enc_feat2
        outputs["equi_enc_feat3"] = equi_enc_feat3
        outputs["equi_enc_feat4"] = equi_enc_feat4
        # outputs["equi_dec_feat4"] = equi_dec_feat4
        # outputs["equi_dec_feat3"] = equi_dec_feat3
        # outputs["equi_dec_feat2"] = equi_dec_feat2
        # outputs["equi_dec_feat1"] = equi_dec_feat1
        # outputs["equi_dec_feat0"] = equi_dec_feat0
        outputs["sph_enc_feat0"] = sph_enc_feat0
        outputs["sph_enc_feat1"] = sph_enc_feat1
        outputs["sph_enc_feat2"] = sph_enc_feat2
        outputs["sph_enc_feat3"] = sph_enc_feat3
        return outputs
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from .convnext import *
import copy
from .blocks import Transformer_Block
from .ViT import miniViT, layers
import torch.nn.functional as F

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, subpixelconvolution, Cube2Equirec, Concat, BiProj, CEELayer, SCEELayer

from collections import OrderedDict
class Transformer_cascade(nn.Module):
    def  __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded
# class Conv3x3(nn.Module):
#     """Layer to pad and convolve input
#     """
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(Conv3x3, self).__init__()
#
#         self.pad = nn.ZeroPad2d(1)
#         self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=bias)
#
#     def forward(self, x):
#         out = self.pad(x)
#         out = self.conv(out)
#         return out
#
#
# class ConvBlock(nn.Module):
#     """Layer to perform a convolution followed by ELU
#     """
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(ConvBlock, self).__init__()
#
#         self.conv = Conv3x3(in_channels, out_channels, bias)
#         self.nonlin = nn.ELU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.nonlin(out)
#         return out
#
#
# def upsample(x):
#     """Upsample input tensor by a factor of 2
#     """
#     return F.interpolate(x, scale_factor=2, mode="nearest")

# def subpixelconvolution(x):
#     ps = nn.PixelShuffle(4)
#     return ps(x)
class sconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.sphere_conv = nn.Sequential(
            nn.Conv2d(in_channels*9, in_channels, kernel_size=1, stride=stride,padding=0, groups=in_channels),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(False)
        )
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        #if h_old != h or w_old != w:
            #deformation = deformation.permute(0, 3, 1, 2)
            #deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            #deformation = deformation.permute(0, 2, 3, 1)
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
        #catx = torch.cat((x_mid, x_left, x_left_up, x_up, x_right_up, x_right, x_right_down, x_down, x_left_down),dim=1)
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
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.maxpool=nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x, lut,lut1):
        # x = self.maxpool(x)
        return self.conv(x,lut,lut1)

class SphFuse_bin(nn.Module):
    """ UniFuse Model: Resnet based Euqi Encoder and Cube Encoder + Euqi Decoder
    """
    def __init__(self, num_layers, equi_h, equi_w, pretrained=False, max_depth=10.0,
                 fusion_type="cee", se_in_fusion=True, nbins=100, min_val=0.1, max_val=10,):
        super(SphFuse_bin, self).__init__()
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
                       "biproj": BiProj,
                       "cee": SCEELayer}
        FusionLayer = Fusion_dict[self.fusion_type]


        # self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)
        # self.equi_dec_convs["trans"] = ConvBlockbn(self.num_ch_enc[4]+256, self.num_ch_enc[4])
        # self.equi_dec_convs["trans1"] = ConvBlockbn(self.num_ch_enc[4]+256, self.num_ch_enc[4])
        self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)#
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        # self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        # self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        # self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        # self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0]*16)

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        # self.projectors = nn.ModuleList(list(self.c2e.values()))
        # self.down_channel = nn.Conv2d(1024, 1024 // 4, kernel_size=1, stride=1, padding=0)
        # self.down_channel1 = nn.Conv2d(1024, 1024 // 4, kernel_size=1, stride=1, padding=0)

        self.conv_out_erp = nn.Sequential(nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0),
                                          nn.Softmax(dim=1))
        self.adaptive_bins_layer = miniViT.mViT(32, n_query_channels=128, patch_size=8,
                                                dim_out=100,
                                                embedding_dim=128, norm='linear')
        # self.sigmoid = nn.Sigmoid()
        # self.transformer = Transformer_cascade(256, 8 * 16, depth=6, num_heads=4)
        # self.transformer1 = Transformer_cascade(256, 8 * 16, depth=6, num_heads=4)
        # self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)


    def forward(self, input_equi_image,lut256,lut64,lut32,lut16,lut8):
        bs, c, erp_h, erp_w = input_equi_image.shape
        equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4 = self.equi_encoder(input_equi_image)
        sph_enc_feat0 = self.inc(input_equi_image, lut256, lut64)
        sph_enc_feat1 = self.conv(sph_enc_feat0, lut64)
        sph_enc_feat2 = self.down1(sph_enc_feat1, lut64, lut32)
        sph_enc_feat3 = self.down2(sph_enc_feat2, lut32, lut16)
        sph_enc_feat4 = self.down3(sph_enc_feat3, lut16, lut8) #b 256 32 64
        # layer4_reshape = self.down_channel(sph_enc_feat4)
        # # layer4_reshape = self.down_channel1(sph_enc_feat4)
        # layer4_reshape = layer4_reshape.permute(0, 2, 3, 1).reshape(bs, 8 * 16, -1)
        # layer4_reshape = self.transformer(layer4_reshape)
        #
        # layer4_reshape = layer4_reshape.permute(0, 2, 1).reshape(bs, -1, 8, 16)
        # layer4 = torch.cat([sph_enc_feat4, layer4_reshape], 1)
        #
        # layer4_reshape1 = self.down_channel1(equi_enc_feat4)
        # layer4_reshape1 = layer4_reshape1.permute(0, 2, 3, 1).reshape(bs, 8 * 16, -1)
        # layer4_reshape1 = self.transformer1(layer4_reshape1)
        #
        # layer4_reshape1 = layer4_reshape1.permute(0, 2, 1).reshape(bs, -1, 8, 16)
        # layer41 = torch.cat([equi_enc_feat4, layer4_reshape1], 1)
        #
        #
        # # euqi image decoding fused with cubemap features
        outputs = {}
        # layer4 = self.equi_dec_convs["trans"](layer4)
        # layer41 = self.equi_dec_convs["trans1"](layer41)
        fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, sph_enc_feat4)
        # fused_feat4 = self.equi_dec_convs["fusion_5"](layer4, layer41)
        equi_x = upsample(self.equi_dec_convs["upconv_5"](fused_feat4))


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
        erp_feature = equi_x

        _, bin_widths_normed_erp, range_attention_maps_erp, queries_erp, feature_map_erp = self.adaptive_bins_layer(
            erp_feature)
        range_attention_maps_erp = F.interpolate(range_attention_maps_erp, (erp_h, erp_w), mode='bilinear')
        out_erp = self.conv_out_erp(range_attention_maps_erp)

        bin_widths_erp = (self.max_val - self.min_val) * bin_widths_normed_erp  # .shape = N, dim_out
        bin_widths_erp = nn.functional.pad(bin_widths_erp, (1, 0), mode='constant', value=self.min_val)

        bin_edges_erp = torch.cumsum(bin_widths_erp, dim=1)

        centers_erp = 0.5 * (bin_edges_erp[:, :-1] + bin_edges_erp[:, 1:])
        n_erp, dout_erp = centers_erp.size()
        centers_erp = centers_erp.view(n_erp, dout_erp, 1, 1)

        pred_global = torch.sum(out_erp * centers_erp, dim=1, keepdim=True)

        # equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        # outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)
        outputs["equi_enc_feat0"] = equi_enc_feat0
        outputs["equi_enc_feat1"] = equi_enc_feat1
        outputs["equi_enc_feat2"] = equi_enc_feat2
        outputs["equi_enc_feat3"] = equi_enc_feat3
        outputs["equi_enc_feat4"] = equi_enc_feat4
        outputs["pred_depth"] = pred_global
        outputs["bin_edges_erp"] = bin_edges_erp
        return outputs
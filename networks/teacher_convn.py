import torch
import torch.nn as nn
import numpy as np
import copy

from .convnext import *
from .layers import Conv3x3, ConvBlock, upsample, subpixelconvolution
from collections import OrderedDict
from .blocks import Transformer_Block

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class Equi_convnext_tea(nn.Module):
    def __init__(self, num_layers, equi_h, equi_w, pretrained=False, max_depth=10.0, **kwargs):
        super(Equi_convnext_tea, self).__init__()


        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        # encoder
        self.equi_encoder = convnext_base(pretrained=True)
        self.num_ch_enc = np.array([128, 128, 256, 512, 1024])#
        # decoder  [16, 32, 64, 128, 256]
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        self.equi_dec_convs = OrderedDict()

        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])#+256

        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0]*16)

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])#self.num_ch_dec[0]
        # self.equi_dec_convs["deconv_0_copy"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])#self.num_ch_dec[0]

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        # self.equi_dec_convs["feature_up_sample"] = ConvBlock(self.num_ch_enc[0], self.num_ch_dec[0])

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))

        # self.down = nn.Conv2d(1024, 1024 // 4, kernel_size=1, stride=1, padding=0)
        # self.transformer = Transformer_cascade(256, 8 * 16, depth=6, num_heads=4)
        self.conv4 = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        self.ln =LayerNorm(128, eps=1e-6, data_format="channels_first")
        self.sigmoid = nn.Sigmoid()
        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)

    def forward(self, input_equi_image):
        bs, c, erp_h, erp_w = input_equi_image.shape
        x = self.conv4(input_equi_image)
        x = self.ln(x)
        equi_enc_feat0 = x
        x = self.equi_encoder.stages[0](x)
        equi_enc_feat1 = x
        x = self.equi_encoder.downsample_layers[1](x)
        x = self.equi_encoder.stages[1](x)
        equi_enc_feat2 = x
        x = self.equi_encoder.downsample_layers[2](x)
        x = self.equi_encoder.stages[2](x)
        equi_enc_feat3 = x
        x = self.equi_encoder.downsample_layers[3](x)
        x = self.equi_encoder.stages[3](x)
        equi_enc_feat4 = x
        # equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4 = self.equi_encoder(input_equi_image)
        outputs = {}


        equi_x = equi_enc_feat4
        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        equi_dec_feat3 = equi_x
        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))#(4,128,64,128)
        equi_dec_feat2 = equi_x
        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)#(4,128,64,128)
        equi_x =self.equi_dec_convs["upconv_2"](equi_x)
        # up = F.interpolate(de_conv0_1, size=(layer2.shape[-2], layer2.shape[-1]), mode='bilinear', align_corners=False)

        equi_dec_feat1 = equi_x
        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)#equi_enc_feat0:4 128 64 128
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = subpixelconvolution(self.equi_dec_convs["upconv_1"](equi_x))#4 32 256 512
        # equi_enc_feat0_32 = self.equi_dec_convs["feature_up_sample"](equi_enc_feat0)
        # equi_x = torch.cat([equi_x, equi_enc_feat0_32], 1)

        equi_dec_feat0 = equi_x
        equi_x = self.equi_dec_convs["deconv_0"](equi_x)
        # equi_x = self.equi_dec_convs["deconv_0_copy"](equi_x)
        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)
        outputs["equi_enc_feat0"] = equi_enc_feat0
        outputs["equi_enc_feat1"] = equi_enc_feat1
        outputs["equi_enc_feat2"] = equi_enc_feat2
        outputs["equi_enc_feat3"] = equi_enc_feat3
        outputs["equi_enc_feat4"] = equi_enc_feat4
        outputs["equi_dec_feat3"] = equi_dec_feat3
        outputs["equi_dec_feat2"] = equi_dec_feat2
        outputs["equi_dec_feat1"] = equi_dec_feat1
        outputs["equi_dec_feat0"] = equi_dec_feat0

        return outputs

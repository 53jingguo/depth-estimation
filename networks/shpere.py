from PIL import Image
# from OpticalFlow_Visualization import flow_vis
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

# import utils
# from vit import ViT
# args = utils.parse_command()
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

class conv1x1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=1)
    def forward(self,x):
        return self.conv(x)

class Sph(nn.Module):
    def __init__(self, n_channels, max_depth=10.0, bilinear=False):
        super(Sph, self).__init__()
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv2(n_channels, 128)
        self.conv = DoubleConv3(128, 128)
        self.down1 = Down(128, 256) #128
        self.down2 = Down(256, 512) #64
        self.down3 = Down(512, 1024) #32

        self.num_ch_enc = np.array([128, 128, 256, 512, 1024])  #
        # decoder  [16, 32, 64, 128, 256]
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        self.equi_dec_convs = OrderedDict()

        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])  # +256

        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0] * 16)

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])  # self.num_ch_dec[0]
        # self.equi_dec_convs["deconv_0_copy"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])#self.num_ch_dec[0]

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        # self.equi_dec_convs["feature_up_sample"] = ConvBlock(self.num_ch_enc[0], self.num_ch_dec[0])

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))

    # self.down = nn.Conv2d(1024, 1024 // 4, kernel_size=1, stride=1, padding=0)
    # self.transformer = Transformer_cascade(256, 8 * 16, depth=6, num_heads=4)

        self.sigmoid = nn.Sigmoid()
        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)
    def deform_input(self, inp, deformation):
        return torch.nn.functional.grid_sample(inp, deformation, align_corners=True)

    def forward(self, x, lut256, lut64, lut32, lut16, lut8):
        equi_enc_feat0 = self.inc(x, lut256, lut64)
        equi_enc_feat1 = self.conv(equi_enc_feat0, lut64)
        equi_enc_feat2 = self.down1(equi_enc_feat1, lut64, lut32)
        equi_enc_feat3 = self.down2(equi_enc_feat2, lut32, lut16)
        equi_enc_feat4 = self.down3(equi_enc_feat3, lut16, lut8) #b 256 32 64
        outputs = {}

        equi_x = equi_enc_feat4
        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))  # (4,128,64,128)

        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)  # (4,128,64,128)
        equi_x = self.equi_dec_convs["upconv_2"](equi_x)
        # up = F.interpolate(de_conv0_1, size=(layer2.shape[-2], layer2.shape[-1]), mode='bilinear', align_corners=False)

        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)  # equi_enc_feat0:4 128 64 128
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = subpixelconvolution(self.equi_dec_convs["upconv_1"](equi_x))  # 4 32 256 512

        # equi_enc_feat0_32 = self.equi_dec_convs["feature_up_sample"](equi_enc_feat0)
        # equi_x = torch.cat([equi_x, equi_enc_feat0_32], 1)

        equi_x = self.equi_dec_convs["deconv_0"](equi_x)
        # equi_x = self.equi_dec_convs["deconv_0_copy"](equi_x)
        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)


        return  outputs
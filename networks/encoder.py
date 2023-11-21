from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample


from collections import OrderedDict

class Equi_en(nn.Module):
    """ Model: Resnet based Encoder + Decoder
    """
    def __init__(self, num_layers, equi_h, equi_w, pretrained=False, max_depth=10.0, **kwargs):
        super(Equi_en, self).__init__()


        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        # encoder
        encoder = {2: mobilenet_v2,
                   18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,
                   152: resnet152}

        if num_layers not in encoder:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        self.equi_encoder = encoder[num_layers](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                           bias=False, padding_mode='zeros')
    def forward(self, depth):
        outputs = {}
        # euqi image encoding
        if self.num_layers < 18:
            equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
                = self.equi_encoder(depth)
        else:
            x = self.conv1(depth)
            x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
            equi_enc_feat0 = x

            x = self.equi_encoder.maxpool(x)
            equi_enc_feat1 = self.equi_encoder.layer1(x)
            equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
            equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
            equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)

        outputs["equi_enc_feat0"] = equi_enc_feat0
        outputs["equi_enc_feat1"] = equi_enc_feat1
        outputs["equi_enc_feat2"] = equi_enc_feat2
        outputs["equi_enc_feat3"] = equi_enc_feat3
        outputs["equi_enc_feat4"] = equi_enc_feat4

        return outputs
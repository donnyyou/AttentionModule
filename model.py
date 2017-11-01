#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math


class Downsampler(nn.Module):
    def __init__(self):
        super(Downsampler, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        out, index = self.pool1(x)
        return out


class BottleNeck(nn.Module):

    def __init__(self, inplanes, planes):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleNeck5(nn.Module):
    def __init__(self, planes):
        super(BottleNeck5, self).__init__()
        self.bottleneck1 = BottleNeck(planes, planes/4)
        self.bottleneck2 = BottleNeck(planes, planes/4)
        self.bottleneck3 = BottleNeck(planes, planes/4)
        self.bottleneck4 = BottleNeck(planes, planes/4)
        self.bottleneck5 = BottleNeck(planes, planes/4)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)

        return x


class Deconv(nn.Module):
    def __init__(self, inplanes, planes):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(inplanes, planes, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x, is_last=False):
        x = self.deconv(x)
        if not is_last:
            x = self.bn(x)
            x = self.relu(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, inplanes, planes):
        super(AttentionModule, self).__init__()
        self.bottleneck1_1 = BottleNeck(inplanes, planes/4)
        self.bottleneck1_2 = BottleNeck(inplanes, planes/4)
        self.downsampler1 = Downsampler()
        self.bottleneck2_1 = BottleNeck(inplanes, planes/4)
        self.downsampler2 = Downsampler()
        self.bottleneck2_2 = BottleNeck(inplanes, planes/4)
        self.bottleneck2_3 = BottleNeck(inplanes, planes/4)
        self.deconv1 = Deconv(inplanes, planes)
        self.bottleneck2_4 = BottleNeck(inplanes, planes/4)
        self.deconv2 = Deconv(inplanes, planes)
        self.conv2_1 = nn.Conv2d(inplanes, planes, 1)
        self.conv2_2 = nn.Conv2d(inplanes, planes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_1 = self.bottleneck1_1(x)
        x_1 = self.bottleneck1_2(x_1)

        x_2 = self.downsampler1(x)
        x_2 = self.bottleneck2_1(x_2)
        x_2 = self.downsampler2(x_2)
        x_2 = self.bottleneck2_2(x_2)
        x_2 = self.bottleneck2_3(x_2)
        x_2 = self.deconv1(x_2)
        x_2 = self.bottleneck2_4(x_2)
        x_2 = self.deconv2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.conv2_2(x_2)
        x_2 = self.sigmoid(x_2)
        x = x_1 * x_2
        x = x + x_1
        return x_1, x


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.downsampler1 = Downsampler()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.downsampler2 = Downsampler() 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.bottleneck_51 = BottleNeck5(128)
        self.downsampler3 = Downsampler()

        self.bottleneck_52 = BottleNeck5(128)

        self.deconv1 = Deconv(128, 64)
        self.bottleneck_1 = BottleNeck(64, 16)
        self.attentionmodule1 = AttentionModule(64, 64)
        self.deconv2 = Deconv(64, 32)
        self.bottleneck_2 = BottleNeck(32, 8)
        self.attentionmodule2 = AttentionModule(32, 32)
        self.deconv3 = Deconv(32, self.num_classes)
        # self.conv_1_1 = nn.Conv2d(16, self.num_classes, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.downsampler1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_1 = self.relu2(x) 
        x = self.downsampler2(x_1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x_2 = self.bottleneck_51(x)
        x = self.downsampler3(x_2)
        x = self.bottleneck_52(x)
        x = self.deconv1(x)
        x = self.bottleneck_1(x)
        b_x1, x1 = self.attentionmodule1(x)

        x = self.deconv2(x1)
        x = self.bottleneck_2(x)
        b_x2, x2 = self.attentionmodule2(x)

        x = self.deconv3(x2, True)
        return b_x1, x1, b_x2, x2, x

if __name__ == "__main__":
    pass

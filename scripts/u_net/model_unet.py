import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

# Based on https://discuss.pytorch.org/t/unet-implementation/426

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, padding=1):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=kernel_size, padding=padding)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False, padding=1):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2, padding=0)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=padding)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):

        up = self.up(x)
        #crop1 = self.center_crop(bridge, up.size()[2])
        #out = torch.cat([up, crop1], 1)

        #print crop1.size()
        #print up.size()
        #print bridge.size()
        out = torch.cat([up, bridge], 1)

        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))


        return out


class UNet(nn.Module):
    def __init__(self, channel_count, class_count):
        super(UNet, self).__init__()
        #self.imsize = imsize

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

        M = 64

        self.conv_block1_64 = UNetConvBlock(channel_count, M)
        self.conv_block64_128 = UNetConvBlock(M, M*2)
        self.conv_block128_256 = UNetConvBlock(M*2, M*4)
        self.conv_block256_512 = UNetConvBlock(M*4, M*8)
        self.conv_block512_1024 = UNetConvBlock(M*8, M*16)

        self.up_block1024_512 = UNetUpBlock(M*16, M*8)
        self.up_block512_256 = UNetUpBlock(M*8, M*4)
        self.up_block256_128 = UNetUpBlock(M*4, M*2)
        self.up_block128_64 = UNetUpBlock(M*2, M)

        self.last = nn.Conv2d(M, class_count, 1)


    def forward(self, x):

        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        block4 = F.dropout(block4, p=0.5, training=self.training)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)
        block5 = F.dropout(block5, p=0.5, training=self.training)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)
        
        out = self.last(up4)

        return out

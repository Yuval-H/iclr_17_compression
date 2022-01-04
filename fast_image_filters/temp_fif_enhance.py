"""
File: custom_nets.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: network architecture for fast image filters
"""
import torch
import torch.nn as nn
#from torchsummary import summary

from fast_image_filters.basic_blocks import ConvBlock


class temp_FIF_enhance(nn.Module):

    """Model architecture for fast image filter"""

    def __init__(self):
        """Initialization """
        super().__init__()

        nbLayers = 64


        self.feature1 = nn.Sequential(
            ConvBlock(3, nbLayers, 3, 1, 1),
            ConvBlock(nbLayers, nbLayers, 3, 2, 2),
            ConvBlock(nbLayers, nbLayers, 3, 8, 8),
            ConvBlock(nbLayers, nbLayers, 3, 16, 16),
            ConvBlock(nbLayers, nbLayers, 3, 32, 32),
            ConvBlock(nbLayers, nbLayers, 3, 64, 64)
        )

        self.feature2 = nn.Sequential(
            ConvBlock(3, nbLayers, 3, 1, 1),
            ConvBlock(nbLayers, nbLayers, 3, 2, 2),
            ConvBlock(nbLayers, nbLayers, 3, 8, 8),
            ConvBlock(nbLayers, nbLayers, 3, 16, 16),
            ConvBlock(nbLayers, nbLayers, 3, 32, 32),
            ConvBlock(nbLayers, nbLayers, 3, 64, 64)
        )

        self.combine_and_rec = nn.Sequential(
            ConvBlock(2*nbLayers, nbLayers, 3, 1, 1),
            nn.Conv2d(nbLayers, 3, kernel_size=1, dilation=1)
        )
    def forward(self, LR_im, HR_SI_im):
        features1 = self.feature1(LR_im)
        features2 = self.feature2(HR_SI_im)
        rec = self.combine_and_rec(torch.cat((features1, features2), 1))

        return rec

    def weights_init(self, m):
        """conv2d Init
        """
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    net = temp_FIF_enhance().cuda()
    #summary(net, input_size=(3, 500, 500))
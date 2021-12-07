
import torch.nn as nn
import torch

import pytorch_msssim

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)



class Cheng2020Attention_reg_0_0625(nn.Module):

    def __init__(self, N=128, **kwargs):
        #super().__init__(N=N, **kwargs)
        super().__init__()

        self.out_channel_N = N
        self.g_a = nn.Sequential(
            ResidualBlock(3, 3),
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.g_a22 = nn.Sequential(
            conv3x3(N, 64, stride=1),
            ResidualBlock(64, 64),
            ResidualBlockWithStride(64, 64, stride=2),
            AttentionBlock(64),
            conv3x3(64, 32, stride=1),
            ResidualBlock(32, 32),
            conv3x3(32, 8, stride=1),
            AttentionBlock(8),
        )

        self.g_s22 = nn.Sequential(
            AttentionBlock(8),
            conv3x3(8, 32, stride=1),
            ResidualBlock(32, 32),
            conv3x3(32, 64, stride=1),  # temp remove
            ResidualBlock(64, 64),  # temp remove
            ResidualBlockUpsample(64, N, 2),
            ResidualBlock(N, N),
        )

        self.g_z1hat_z2 = nn.Sequential(
            AttentionBlock(2 * N),
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
        )


    def forward(self, im1, im2):

        channels = 8
        quant_noise_feature2 = torch.zeros(im1.size(0), channels, im1.size(2) // 32, im1.size(3) // 32).cuda()
        quant_noise_feature2 = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature2), -8, 8)

        z1 = self.g_a(im1)
        z2 = self.g_a(im2)
        # further compress z1:
        if self.training:
            z1_down = self.g_a22(z1) + quant_noise_feature2
        else:
            z1_down = torch.round(self.g_a22(z1)/16)*16

        # clamp it to 8 bits - THIS IS THE COMPRESSED REPRESENTATION - WILL BE SENT OVER CHANNEL
        z1_down = torch.clamp(z1_down, -128, 128)

        z1_hat = self.g_s22(z1_down)

        # cat z1_hat, z2 -> get z1_hat_hat
        z_cat = torch.cat((z1_hat, z2), 1)
        # z_cat = torch.cat((torch.zeros_like(z1_hat), z2), 1)
        # z_cat = torch.cat((z1_hat, torch.zeros_like(z2)), 1)


        z1_hat_hat = self.g_z1hat_z2(z_cat)

        # recon images
        final_im1_recon = self.g_s(z1_hat_hat)

        if self.training:
            return final_im1_recon
        else:
            return final_im1_recon, z1_down


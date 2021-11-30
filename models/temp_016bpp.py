
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



class Cheng2020Attention_0_16bpp(nn.Module):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels (base auto-encoder)

    """

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
            ResidualBlock(64, 41),
            ResidualBlock(41, 41),
            AttentionBlock(41)
        )

        self.g_s22 = nn.Sequential(
            AttentionBlock(41),
            ResidualBlock(41, 41),
            ResidualBlock(41, 64),
            ResidualBlock(64, 64),
            ResidualBlockUpsample(64, N, 2),
            ResidualBlock(N, N),
        )

        self.g_z1hat_z2 = nn.Sequential(
            AttentionBlock(2*N),
            ResidualBlock(2*N, 2*N),
            ResidualBlock(2*N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
        )


    def forward(self, im1, im2):
        quant_noise_feature = torch.zeros(im1.size(0), self.out_channel_N, im1.size(2) // 16,
                                          im1.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -8, 8)

        channels = 41
        quant_noise_feature2 = torch.zeros(im1.size(0), channels, im1.size(2) // 32, im1.size(3) // 32).cuda()
        quant_noise_feature2 = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature2), -8, 8)

        z1 = self.g_a(im1)
        z2 = self.g_a(im2)
        if self.training:
            compressed_z1 = z1 + quant_noise_feature
            compressed_z2 = z2 + quant_noise_feature
        else:
            compressed_z1 = torch.round(z1)
            compressed_z2 = torch.round(z2)

        # further compress z1
        if self.training:
            z1_down = self.g_a22(z1) + quant_noise_feature2
        else:
            z1_down = torch.round(self.g_a22(z1)/16)*16

        # clamp it to 8 bits
        z1_down = torch.clamp(z1_down, -128, 128)

        z1_hat = self.g_s22(z1_down)

        # cat z1_hat, z2 -> get z1_hat_hat
        z_cat = torch.cat((z1_hat, z2), 1)
        #z_cat = torch.cat((torch.zeros_like(z1_hat), z2), 1)
        #z_cat = torch.cat((z1_hat, torch.zeros_like(z2)), 1)

        z1_hat_hat = self.g_z1hat_z2(z_cat)

        # recon images:
        final_im1_recon = self.g_s(z1_hat_hat)

        im1_hat = self.g_s(compressed_z1)
        im2_hat = self.g_s(compressed_z2)

        # distortion
        useL1 = False
        use_msssim = True
        if useL1:
            #loss = torch.mean(torch.sqrt((diff * diff)
            loss_l1 = nn.L1Loss()

            mse_loss = 0.5 * loss_l1(im1_hat.clamp(0., 1.), im1) + 0.5 * loss_l1(im2_hat.clamp(0., 1.), im2)
            mse_on_z = loss_l1(z1_hat_hat, z1)
            mse_on_full = loss_l1(final_im1_recon.clamp(0., 1.), im1)
        elif use_msssim:
            mse_loss = 1 - (0.5*(pytorch_msssim.ms_ssim( final_im1_recon.clamp(0., 1.), im1, data_range=1.0) +
                            pytorch_msssim.ms_ssim(im2_hat.clamp(0., 1.), im2, data_range=1.0)))
            mse_on_z = 1
            mse_on_full = 1 - pytorch_msssim.ms_ssim(final_im1_recon.clamp(0., 1.), im1, data_range=1.0)
        else:
            mse_loss = 0.5*torch.mean((im1_hat.clamp(0., 1.) - im1).pow(2)) + 0.5*torch.mean((im2_hat.clamp(0., 1.) - im2).pow(2))
            mse_on_z = torch.mean((z1_hat_hat - z1).pow(2))
            mse_on_full = torch.mean((final_im1_recon.clamp(0., 1.) - im1).pow(2))

        if self.training:
            return mse_loss, mse_on_full, mse_on_z, torch.clip(final_im1_recon, 0, 1)
        else:
            return mse_loss, mse_on_full, torch.clip(final_im1_recon, 0, 1), z1_down

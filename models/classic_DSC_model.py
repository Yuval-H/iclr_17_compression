# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

from compressai.entropy_models import EntropyBottleneck, EntropyModel

from compressai.models.priors import JointAutoregressiveHierarchicalPriors


class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class classic_DSC_model(nn.Module): #(Cheng2020Anchor):
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

        self.use_another_net_on_recon = False
        self.out_channel_N = N
        self.g_a = nn.Sequential(
            ResidualBlock(3, 3),                        ## YH patch : remove to use the pretrained weights
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            #AttentionBlock(N),   ### added
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_a_Y = nn.Sequential(
            ResidualBlock(3, 3),  ## YH patch : remove to use the pretrained weights
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            # AttentionBlock(N),   ### added
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            #AttentionBlock(N), ### added
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            #AttentionBlock(N),  ### added
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.g_a22 = nn.Sequential(
            #conv3x3(128, 64, stride=1),
            ResidualBlock(128, 64),             ## YH Patch: this is added (instead of the upper line)
            #AttentionBlock(64),  ### added
            ResidualBlock(64, 64),
            ResidualBlockWithStride(64, 64, stride=2),
            AttentionBlock(64),
            #conv3x3(64, 32, stride=1),
            ResidualBlock(64, 32),  ## YH Patch: this is added (instead of the upper line)
            #AttentionBlock(32),  ### added
            ResidualBlock(32, 32),
            #conv3x3(32, 8, stride=1),
            ResidualBlock(32, 8),  ## YH Patch: this is added (instead of the upper line)
            AttentionBlock(8),
        )

        self.g_s22 = nn.Sequential(
            AttentionBlock(8),
            #conv3x3(8, 32, stride=1),
            ResidualBlock(8, 32),  ## YH Patch: this is added (instead of the upper line)
            ResidualBlock(32, 32),
            #conv3x3(32, 64, stride=1),
            ResidualBlock(32, 64),  ## YH Patch: this is added (instead of the upper line)
            #AttentionBlock(64),  ### added
            ResidualBlock(64, 64),
            #AttentionBlock(64),
            ResidualBlockUpsample(64, 128, 2),
            #AttentionBlock(128),  ### added
            ResidualBlock(128, 128),
        )

        self.g_z1hat_z2 = nn.Sequential(
            AttentionBlock(256),
            ResidualBlock(256, 256),
            #AttentionBlock(256),  ### added
            ResidualBlock(256, 128),
            AttentionBlock(128),
            ResidualBlock(128, 128),
            #AttentionBlock(128),  ### added
        )

        self.g_rec1_im2 = nn.Sequential(
            AttentionBlock(6),
            ResidualBlock(6, 6),
            AttentionBlock(6),  ### added
            ResidualBlock(6, 3),
            ##ResidualBlock(3, 3),
            AttentionBlock(3),
            ResidualBlock(3, 3),
            AttentionBlock(3),  ### added
        )

        self.g_rec1_im2_new = nn.Sequential(
            AttentionBlock(6),
            ResidualBlock(6, 3),
            ResidualBlock(3, 3),
            AttentionBlock(3),
            ResidualBlock(3, 3),
        )

    def forward(self, im1, im2):
        quant_noise_feature = torch.zeros(im1.size(0), self.out_channel_N, im1.size(2) // 16,
                                          im1.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)

        quant_noise_feature2 = torch.zeros(im1.size(0), 8, im1.size(2) // 32,
                                          im1.size(3) // 32).cuda()
        quant_noise_feature2 = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature2), -0.5, 0.5)

        z1 = self.g_a(im1)
        z2 = self.g_a_Y(im2)

        # further compress z1
        if self.training:
            z1_down = self.g_a22(z1) + quant_noise_feature2
        else:
            z1_down = torch.round(self.g_a22(z1))

        z1_hat = self.g_s22(z1_down)

        # cat z1_hat, z2 -> get z1_hat_hat
        z_cat = torch.cat((z1_hat, z2), 1)
        #z_cat = torch.cat((torch.zeros_like(z1_hat), z2), 1)
        #z_cat = torch.cat((z1_hat, torch.zeros_like(z2)), 1)
        z1_hat_hat = self.g_z1hat_z2(z_cat)

        # recon images
        final_im1_recon = self.g_s(z1_hat_hat)

        if self.use_another_net_on_recon:
            # Note: adding the net results as a residual to the reconstructed image.
            cat_rec_and_im2 = torch.cat((final_im1_recon, im2), 1)
            final_im1_recon = final_im1_recon + self.g_rec1_im2_new(cat_rec_and_im2)


        # distortion
        useL1 = True
        if useL1:
            #loss = torch.mean(torch.sqrt((diff * diff)
            loss_l1 = nn.L1Loss()

            mse_on_z = loss_l1(z1_hat_hat, z1)
            mse_on_full = loss_l1(final_im1_recon.clamp(0., 1.), im1)
        else:
            mse_on_z = torch.mean((z1_hat_hat - z1).pow(2))
            mse_on_full = torch.mean((final_im1_recon.clamp(0., 1.) - im1).pow(2))

        if self.training:
            return 1, mse_on_full, mse_on_z
        else:
            return 1, mse_on_full, torch.clip(final_im1_recon, 0, 1), z1_down

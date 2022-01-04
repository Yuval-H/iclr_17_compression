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

import pytorch_msssim

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)



class finalEnhanceNet(nn.Module):


    def __init__(self, N=64, **kwargs):
        super().__init__()

        self.conv_a = nn.Sequential(
            ResidualBlock(6, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
        )

        self.conv_b = nn.Sequential(
            ResidualBlock(6, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            nn.Conv2d(N, N, kernel_size=1, stride=1)
        )

        self.final_block = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlock(N, 3),
        )

    def forward(self, im_rec_SI_cat):
        a = self.conv_a(im_rec_SI_cat)
        b = self.conv_b(im_rec_SI_cat)
        out = a * torch.sigmoid(b)

        out = self.final_block(out)
        return out


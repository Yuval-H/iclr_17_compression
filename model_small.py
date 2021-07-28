import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *
from models.analysis_small import Analysis_small_net
from models.synthesis_small import Synthesis_small_net

def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, **kwargs)
    nn.init.kaiming_normal_(_conv.weight)
    nn.init.zeros_(_conv.bias)
    return _conv

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor_small(nn.Module):
    def __init__(self, out_channel_N=320):
        super(ImageCompressor_small, self).__init__()
        #self.Encoder = Analysis_small_net()
        #self.Decoder = Synthesis_small_net()
        self.out_channel_N = out_channel_N

        self.attention_layer = nn.MultiheadAttention

        self.conv_down_zx = nn.Sequential(conv(128, 64, 3, 1, 1), nn.ReLU(),
                                          conv(64, 64, 1, 1, 0), nn.ReLU(),
                                          conv(64, 32, 3, 1, 1), nn.ReLU(),
                                          conv(32, 32, 1, 1, 0), nn.ReLU())
        #self.fc_down_zx = nn.Sequential(nn.Linear(8192, 1024), nn.ReLU())

        self.fc_combine_zx_zy = nn.Sequential(conv(256, 256, 7, 1, 3),
                                          conv(256, 256, 7, 1, 3),
                                          conv(256, 128, 3, 1, 1),
                                          conv(128, 128, 3, 1, 1),
                                          conv(128, 128, 3, 1, 1))


    def forward(self, z1, z2):

        batch_size = z1.size()[0]
        z1_down = self.conv_down_zx(z1)
        #z1_down = self.fc_down_zx(z1_down.view(batch_size, -1))

        feature1 = z1 # 2048 length
        feature2 = z2

        #feature1 = feature1.view(batch_size, -1)
        #feature2 = feature2.view(batch_size, -1)
        feature_cat = torch.cat((feature1, feature2), 1)  # length of 8192

        recon_z = self.fc_combine_zx_zy(feature_cat)
        recon_z = recon_z.view(z1.size())
        # recon_image = prediction + recon_res
        # distortion
        mse_loss = torch.mean((recon_z - z1).pow(2))


        return recon_z, feature1, mse_loss

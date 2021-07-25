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
from models.analysis_17_new import Analysis_net_17_new


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


class ImageCompressor_new(nn.Module):
    def __init__(self, out_channel_N=128):
        super(ImageCompressor_new, self).__init__()
        #self.Encoder = Analysis_net_17_new(out_channel_N=out_channel_N)
        self.Encoder = Analysis_net_17_new(out_channel_N=out_channel_N)
        self.Decoder = Synthesis_net_17(out_channel_N=out_channel_N)
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        self.out_channel_N = out_channel_N

    def forward(self, input_image):
        feature, feature_before_thresh = self.Encoder(input_image)
        feature_renorm = feature
        #quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(feature), -0.7, 0.7)
        if self.training:
            compressed_feature_renorm = feature_renorm #+ quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        #mse_loss = torch.mean((recon_image - input_image).pow(2))


        return clipped_recon_image, compressed_feature_renorm, feature_before_thresh
        #return clipped_recon_image, compressed_feature_renorm

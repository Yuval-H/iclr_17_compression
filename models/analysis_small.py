#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
#from .basics import *
import pickle
import os
import codecs
import torch.nn as nn
import torch
from .GDN import GDN
import math
from models.binarizer import *


class Analysis_small_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=512, out_channel_M=16):
        super(Analysis_small_net, self).__init__()
        self.conv1 = nn.Conv2d(1024, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.fc1 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU())
        self.fc2 = nn.Linear(2048, 1024)
        self.head = nn.Sequential(nn.Sigmoid(), Lambda(bin_values))

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        x = self.fc1(x.view(x.size()[0], -1))
        x = self.fc2(x)
        return x  # self.head(x)


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_small_net()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()

#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
import pickle
import os
import codecs
from .analysis import Analysis_net

class Synthesis_small_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=512, out_channel_M=16):
        super(Synthesis_small_net, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU())
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 1, stride=1, padding=0, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3, stride=1, padding=1, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 1, stride=1, padding=0, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 1024, 3, stride=1, padding=1, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

        self.conv1 = nn.Sequential(nn.Conv2d(2048, 1024, 7, stride=1, padding=3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(1024, 1024, 7, stride=1, padding=3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(1024, 1024, 7, stride=1, padding=3), nn.Sigmoid())


    def forward(self, features_compressed, z2):
        x = self.fc1(features_compressed)
        x = self.fc2(x)
        # expand dims to fit conv
        x = x[...,None,None]
        x = x.view(x.size()[0], 16, 16, 16)
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        # concatenate feature and calc Z_hat
        feature_cat = torch.cat((x, z2), 1)
        z_hat = self.conv1(feature_cat)
        z_hat = self.conv2(z_hat)
        z_hat = self.conv3(z_hat)

        return x#z_hat

# synthesis_one_pass = tf.make_template('synthesis_one_pass', synthesis_net)

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    synthesis_net = Synthesis_net()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

# def main(_):
#   build_model()


if __name__ == '__main__':
    build_model()

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import PIL
import glob
import numpy as np
from model_new import *
from model import *
from models.temp import Cheng2020Attention
from models.temp_and_FIF import Cheng2020Attention_FIF
from models.temp_1bpp import Cheng2020Attention_1bpp
from models.temp_016bpp import Cheng2020Attention_0_16bpp
from models.temp_highBitRate import Cheng2020Attention_highBitRate2
from models.test_freqSepNet import Cheng2020Attention_freqSep
import gzip
import pytorch_msssim

from utils.Conditional_Entropy import compute_conditional_entropy
#/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/0_16bpp net/model_bestVal_loss0.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = Cheng2020Attention_FIF()
#model = Cheng2020Attention_1bpp()
#model = Cheng2020Attention_freqSep()
model = Cheng2020Attention_0_16bpp()
#model = Cheng2020Attention()
#model = Cheng2020Attention_highBitRate2()


checkpoint = torch.load(pretrained_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview/testing/image_2'
stereo2_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview/testing/image_2'


# smaller dataset:
#stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_2'

# CLIC view test:
#stereo1_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im3'
#stereo2_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im2'

list1 = glob.glob(os.path.join(stereo1_dir, '*11.png'))
list2 = glob.glob(os.path.join(stereo2_dir, '*11.png'))
stereo1_path_list = list1 + list2
#stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*.png'))
#stereo1_path_list = glob.glob(os.path.join('/home/access/dev/Holopix50k/test/left', '*.jpg'))


transform = transforms.Compose([transforms.CenterCrop((320, 1216)), transforms.ToTensor()])



n_channels = 41
mask1 = 31
mask2 = 16
best_channel = -1
best_msssim = 0

for channel in range(n_channels):
    if channel == mask1 or channel == mask2:
        continue
    avg_bpp = 0
    avg_psnr = 0
    avg_msssim = 0
    for i in range(len(stereo1_path_list)):
        img_stereo1 = Image.open(stereo1_path_list[i])
        #img_stereo2_name = stereo1_path_list[i].replace('left', 'right')
        img_stereo2_name = stereo1_path_list[i].replace('image_2', 'image_3')
        img_stereo2 = Image.open(img_stereo2_name)
        img_stereo1 = transform(img_stereo1)
        img_stereo2 = transform(img_stereo2)
        ##
        input1 = img_stereo1[None, ...].to(device)
        input2 = img_stereo2[None, ...].to(device)
        # Encoded images:
        mse_loss, mse_on_full, final_im1_recon, z1_down = model(input1, input2, mask_channels=[mask1, mask2, channel])
        numpy_input_image = img_stereo1.permute(1, 2, 0).detach().numpy()
        tensor_output_image = torch.squeeze(final_im1_recon).permute(1, 2, 0)
        numpy_output_image = tensor_output_image.cpu().detach().numpy()
        mse = np.mean(np.square(numpy_input_image - numpy_output_image))  # * 255**2   #mse_loss.item()/2
        psnr = -20*np.log10(np.sqrt(mse))
        msssim = pytorch_msssim.ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0)
        e1 = torch.squeeze(z1_down.cpu()).detach().numpy().flatten()
        e1 = (e1 +128).astype(np.uint8)
        n_bits = gzip.compress(e1).__sizeof__() * 8
        n_pixel = numpy_input_image.shape[0]*numpy_input_image.shape[1]
        bpp = n_bits/n_pixel

        avg_bpp += bpp
        avg_msssim += msssim.item()
        avg_psnr += psnr


    count = len(stereo1_path_list)
    avg_bpp = avg_bpp / count
    avg_psnr = avg_psnr / count
    avg_msssim = avg_msssim / count
    print('Mask channel', channel, ',  msssim, psnr, bpp = ', avg_msssim, avg_psnr, avg_bpp)

    if avg_msssim > best_msssim:
        best_msssim = avg_msssim
        best_channel = channel

print('the best channel to remove is - ', best_channel)
print('with the channels masked we got msssim = ', best_msssim)




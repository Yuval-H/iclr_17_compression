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
save_img_and_recon_for_GPNN = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = Cheng2020Attention_1bpp()
#model = Cheng2020Attention_0_16bpp()
model1 = Cheng2020Attention_highBitRate2()
#model2 = Cheng2020Attention_highBitRate2()
#model = Cheng2020Attention_highBitRate2()

pretrained_model_path1 = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/ABLATION/0.0625bpp/model_best_weights (1).pth'
#pretrained_model_path2 = ''

checkpoint = torch.load(pretrained_model_path1)
model1.load_state_dict(checkpoint['model_state_dict'])
#checkpoint = torch.load(pretrained_model_path2)
#model2.load_state_dict(checkpoint['model_state_dict'])

model1 = model1.to(device)
model1.eval()
#model2 = model2.to(device)
#model2.eval()

stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview/testing/image_2'#'/home/access/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview/testing/image_2'
#stereo2_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview/testing/image_2'#'/home/access/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview/testing/image_2'


# smaller dataset:
#stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_2'
#stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_03'
#stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_3_OF_to_2'

# CLIC view test:
#stereo1_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im3'
#stereo2_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im2'

#list1 = glob.glob(os.path.join(stereo1_dir, '*11.png'))
#list2 = glob.glob(os.path.join(stereo2_dir, '*11.png'))
#stereo1_path_list = list1 + list2
stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*.png'))
#stereo1_path_list = glob.glob(os.path.join('/home/access/dev/Holopix50k/test/left', '*.jpg'))


#transform = transforms.Compose([transforms.Resize((192, 608), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])
transform = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((360, 360)), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((320, 960), interpolation=Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((370, 740)),transforms.Resize((128, 256), interpolation=3), transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor()])



for i in range(len(stereo1_path_list)):
    img_stereo1 = Image.open(stereo1_path_list[i])
    #img_stereo2_name = stereo1_path_list[i].replace('left', 'right')
    img_stereo2_name = stereo1_path_list[i].replace('image_2', 'image_3')
    img_stereo2 = Image.open(img_stereo2_name)
    input1 = transform(img_stereo1)
    input2 = transform(img_stereo2)
    # cut image H*W to be a multiple of 16
    M = 32
    shape = input1.size()
    input1 = input1[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    input2 = input2[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    ##
    input1 = input1[None, ...].to(device)
    input2 = input2[None, ...].to(device)

    # Decoded images of two models, use average to reduce noise
    _, _, final_im1_recon1, z1_down = model1(input1, input2)
    #_, _, final_im1_recon2, _ = model2(input1, input2)

    numpy_input_image = torch.squeeze(input1).permute(1, 2, 0).cpu().detach().numpy()
    numpy_output_image1 = torch.squeeze(final_im1_recon1).permute(1, 2, 0).cpu().detach().numpy()
    #numpy_output_image2 = torch.squeeze(final_im1_recon2).permute(1, 2, 0).cpu().detach().numpy()
    '''
    diff1 = numpy_input_image - numpy_output_image1
    diff1 = (diff1 - diff1.min())/(diff1.max() - diff1.min())
    #diff2 = 128 + (original - numpy_output_image2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #fig.suptitle('Original, Diff - 0.03_bpp, 0.06_bpp')
    fig.suptitle('Original, Recon, Diff (0.06_bpp Ablation)')
    ax1.imshow(numpy_input_image)
    ax2.imshow(numpy_output_image1)
    ax3.imshow(diff1)
    fig.tight_layout()

    plt.show()
    '''

    if True:
        _, _, final_im1_recon2, z1_down = model1(input2, input2)
        numpy_output_SI = torch.squeeze(final_im1_recon2).permute(1, 2, 0).cpu().detach().numpy()

        orig_path =     '/media/access/SDB500GB/dev/data_sets/kitti/Ablation/original/'
        orig_si_path = '/media/access/SDB500GB/dev/data_sets/kitti/Ablation/SI/'
        rec_path =      '/media/access/SDB500GB/dev/data_sets/kitti/Ablation/recon-original/'
        rec_SI_path =   '/media/access/SDB500GB/dev/data_sets/kitti/Ablation/recon-SI/'
        orig_si_numpy = torch.squeeze(input2).permute(1, 2, 0).cpu().detach().numpy()
        orig_si_image = Image.fromarray((orig_si_numpy*255).astype(np.uint8))
        orig_image = Image.fromarray((numpy_input_image*255).astype(np.uint8))
        rec_SI_img = Image.fromarray((numpy_output_SI*255).astype(np.uint8))
        rec_img = Image.fromarray((numpy_output_image1*255).astype(np.uint8))

        orig_si_image.save(orig_si_path+stereo1_path_list[i][-13:])
        orig_image.save(orig_path+stereo1_path_list[i][-13:])
        rec_SI_img.save(rec_SI_path+stereo1_path_list[i][-13:])
        rec_img.save(rec_path + stereo1_path_list[i][-13:])





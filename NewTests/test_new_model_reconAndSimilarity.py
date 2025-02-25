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
from models.temp_att_0_03bpp import Cheng2020Attention_ATT
from models.temp_1bpp import Cheng2020Attention_1bpp
from models.temp_016bpp import Cheng2020Attention_0_16bpp
from models.temp_highBitRate import Cheng2020Attention_highBitRate2
#from models.test_freqSepNet import Cheng2020Attention_freqSep
from models.temp_bottleneck_Att import Cheng2020Attention_1bpp_Att
import gzip
import pytorch_msssim

from utils.Conditional_Entropy import compute_conditional_entropy
#/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow
save_img_and_recon_for_GPNN = False
load_model_new_way = True
pretrained_model_path = '/media/access/SDB500GB/dev/iclr_17_compression/ckpoints_newest/checkpoints_new/new_net/Sharons dataset/1bpp net (0.125bpp)/model_bestVal_loss_fullMSSSIM.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = Cheng2020Attention_FIF()
model = Cheng2020Attention_1bpp()
#model = Cheng2020Attention_ATT()
#model = Cheng2020Attention_freqSep()
#model = Cheng2020Attention_0_16bpp()
#model = Cheng2020Attention()
#model = Cheng2020Attention_highBitRate2()
#model = Cheng2020Attention_1bpp_Att()


if load_model_new_way:
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()

stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview/testing/image_2'#'/home/access/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview/testing/image_2'
stereo2_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview/testing/image_2'#'/home/access/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview/testing/image_2'
#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_2'
#stereo2_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_3'

# smaller dataset:
stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_2'
#stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_03'
#stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_3_OF_to_2'

# CLIC view test:
#stereo1_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im3'
#stereo2_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im2'

list1 = glob.glob(os.path.join(stereo1_dir, '*11.png'))
list2 = glob.glob(os.path.join(stereo2_dir, '*11.png'))
#stereo1_path_list = list1 + list2
stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*.png'))
#stereo1_path_list = glob.glob(os.path.join('/home/access/dev/Holopix50k/test/left', '*.jpg'))


#transform = transforms.Compose([transforms.Resize((192, 608), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])
transform = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((320, 960)), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((360, 360)), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((320, 960), interpolation=Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((370, 740)),transforms.Resize((128, 256), interpolation=3), transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor()])



avg_bpp = 0
avg_mse = 0
avg_psnr = 0
avg_l1 = 0
avg_msssim = 0
min_mse = 1000
max_mse = 0
min_idx = 0
max_idx = 0
count = 0
temp_min = 0
temp_max = 0

max2_idx = 0
max3_idx = 0


#file_msssim = open('/home/access/dev/DSIN/sharons code/ToSend/images/5.values_list/HoloPix50k/twoStepsCompression/msssim_list_HoloPix50k_target_0.125bpp_twoStepsNet.txt',"a")
#file_bpp = open('/home/access/dev/DSIN/sharons code/ToSend/images/5.values_list/HoloPix50k/twoStepsCompression/bpp_list_HoloPix50k_target_0.125bpp_twoStepNet.txt',"a")

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

    #temp = input1
    #input1 = input2
    #input2 = temp

    # Encoded images:
    mse_loss, mse_on_full, final_im1_recon, z1_down = model(input1, input2)#,mask_channels=[16, 31, 3])  # try to run only with mse_on_full
    numpy_input_image = torch.squeeze(input1).permute(1, 2, 0).cpu().detach().numpy()
    numpy_output_image = torch.squeeze(final_im1_recon).permute(1, 2, 0).cpu().detach().numpy()
    l1 = np.mean(np.abs(numpy_input_image - numpy_output_image))
    mse = np.mean(np.square(numpy_input_image - numpy_output_image))  # * 255**2   #mse_loss.item()/2
    psnr = -20*np.log10(np.sqrt(mse))
    #msssim = ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0, size_average=True, win_size=11) ## should be 11 for full size, 7 for small
    msssim = pytorch_msssim.ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0)
    if not msssim == msssim:
        print('1')
    e1 = torch.squeeze(z1_down.cpu()).detach().numpy().flatten()
    temp_min = np.min((temp_min, e1.min()))
    temp_max = np.max((temp_max, e1.max()))

    if save_img_and_recon_for_GPNN:
        orig_path = '/media/access/SDB500GB/holopix50k/net results/0.031 bpp/original/'
        orig_si_path = '/media/access/SDB500GB/holopix50k/net results/0.031 bpp/SI/'
        rec_path = '/media/access/SDB500GB/holopix50k/net results/0.031 bpp/reconstructed/'
        orig_si_numpy = torch.squeeze(input2).permute(1, 2, 0).cpu().detach().numpy()
        orig_si_image = Image.fromarray((orig_si_numpy*255).astype(np.uint8))
        orig_image = Image.fromarray((numpy_input_image*255).astype(np.uint8))
        rec_img = Image.fromarray((numpy_output_image*255).astype(np.uint8))
        orig_si_image.save(orig_si_path+stereo1_path_list[i][-13:])
        orig_image.save(orig_path+stereo1_path_list[i][-13:])
        rec_img.save(rec_path+stereo1_path_list[i][-13:])
    '''
    if e1.min() < -128:
        error_message = "code min value is less than 127"
        raise ValueError(error_message)
    if e1.max() > 128:
        error_message = "code max value is more than 127"
        raise ValueError(error_message)
    '''
    e1 = (e1 +128).astype(np.uint8)
    n_bits = gzip.compress(e1).__sizeof__() * 8
    n_pixel = numpy_input_image.shape[0]*numpy_input_image.shape[1]
    bpp = n_bits/n_pixel

    #file_msssim.write(str(msssim.item()) + '\n')
    #file_bpp.write('0.125' + '\n')
    #file_bpp.write(str(bpp) + '\n')

    print(psnr, msssim, bpp)
    avg_bpp += bpp
    avg_msssim += msssim.item()
    avg_l1 = avg_l1 + l1
    avg_mse = avg_mse + mse
    avg_psnr += psnr
    count = count + 1
    if mse > max_mse:
        max_mse = mse
        max3_idx = max2_idx
        max2_idx = max_idx
        max_idx = i
    if mse < min_mse:
        min_mse = mse
        min_idx = i

#file_msssim.close()
#file_bpp.close()

avg_bpp = avg_bpp / count
avg_mse = avg_mse / count
avg_psnr = avg_psnr / count
avg_l1 = avg_l1 / count
avg_msssim = avg_msssim / count
rms = np.sqrt(avg_mse)
print('min  MSE = ', min_mse)
print('max MSE = ', max_mse)
print('average MSE: ', avg_mse,',  ', avg_mse*255**2)
print('average RMS: ', rms,', ', rms*255)
print('average PSNR: ', avg_psnr)
print('average MS-SSIM: ', avg_msssim)
print('average  bpp = ', avg_bpp)


plot_best_and_worst = True
if plot_best_and_worst:
    img1_minDist = Image.open(stereo1_path_list[min_idx])
    img2_minDist = Image.open(stereo1_path_list[min_idx].replace('image_2', 'image_3'))
    in1 = transform(img1_minDist)
    in2 = transform(img2_minDist)
    shape = in1.size()
    in1 = in1[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    in2 = in2[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    img1_minDist = torch.squeeze(in1).permute(1, 2, 0).cpu().detach().numpy()
    input1 = in1[None, ...].to(device)
    input2 = in2[None, ...].to(device)
    # Encoded images:
    _, _, final_im1_recon, _ = model(input1, input2)
    img1_minDist_rec = torch.squeeze(final_im1_recon).permute(1, 2, 0).cpu().detach().numpy()


    img1_maxDist = Image.open(stereo1_path_list[max_idx])
    img2_maxDist = Image.open(stereo1_path_list[max_idx].replace('image_2', 'image_3'))
    ##
    in1 = transform(img1_maxDist)
    in2 = transform(img2_maxDist)
    shape = in1.size()
    in1 = in1[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    in2 = in2[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    img1_maxDist = torch.squeeze(in1).permute(1, 2, 0).cpu().detach().numpy()
    input1 = in1[None, ...].to(device)
    input2 = in2[None, ...].to(device)
    # Encoded images:
    _, _, final_im1_recon, _ = model(input1, input2)
    img1_maxDist_rec = torch.squeeze(final_im1_recon).permute(1, 2, 0).cpu().detach().numpy()
    #img1_maxDist_rec = img1_maxDist
    ##
    print('the worst image is',stereo1_path_list[max_idx])
    print(stereo1_path_list[max2_idx])
    print(stereo1_path_list[max3_idx])


    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Best MSE reconstruction,')
    ax1.imshow(img1_minDist)
    ax2.imshow(img1_minDist_rec)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Worst MSE reconstruction,')
    ax1.imshow(img1_maxDist)
    ax2.imshow(img1_maxDist_rec)
    fig.tight_layout()

    plt.show()

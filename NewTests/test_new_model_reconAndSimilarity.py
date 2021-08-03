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
from model_small import ImageCompressor_small
from models.temp import Cheng2020Attention
import gzip

from utils.Conditional_Entropy import compute_conditional_entropy


pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/using 22 nets on latent/try_L1/iter_111.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Cheng2020Attention()
global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()


#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
#stereo2_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_3'

# smaller dataset:
stereo1_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_03'

# CLIC view test:
#stereo1_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im3'
#stereo2_dir = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/im2'

stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))



#transform = transforms.Compose([transforms.Resize((192, 624), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])


avg_bpp = 0
avg_mse = 0
avg_l1 = 0
avg_msssim = 0
min_mse = 1000
max_mse = 0
min_idx = 0
max_idx = 0
count = 0

for i in range(len(stereo1_path_list)):
    img_stereo1 = Image.open(stereo1_path_list[i])
    img_stereo2_name = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[i]))
    img_stereo2 = Image.open(img_stereo2_name)
    img_stereo1 = transform(img_stereo1)
    img_stereo2 = transform(img_stereo2)
    # cut image H*W to be a multiple of 16
    M = 32
    shape = img_stereo1.size()
    img_stereo1 = img_stereo1[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    img_stereo2 = img_stereo2[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    ##
    input1 = img_stereo1[None, ...].to(device)
    input2 = img_stereo2[None, ...].to(device)

    # Encoded images:
    mse_loss, mse_on_full, final_im1_recon, z1_down = model(input1, input2)

    numpy_input_image = img_stereo1.permute(1, 2, 0).detach().numpy()
    tensor_output_image = torch.squeeze(final_im1_recon).permute(1, 2, 0)
    numpy_output_image = tensor_output_image.cpu().detach().numpy()
    l1 = np.mean(np.abs(numpy_input_image - numpy_output_image))
    mse = np.mean(np.square(numpy_input_image - numpy_output_image))  # * 255**2   #mse_loss.item()/2
    msssim = ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0, size_average=True)

    e1 = torch.squeeze(z1_down.cpu()).detach().numpy().flatten()
    n_bits = gzip.compress(e1).__sizeof__() * 8
    n_pixel = numpy_input_image.shape[0]*numpy_input_image.shape[1]
    bpp = n_bits/n_pixel


    print(mse, msssim, bpp)
    avg_bpp += bpp
    avg_msssim += msssim
    avg_l1 = avg_l1 + l1
    avg_mse = avg_mse + mse
    count = count + 1
    if mse > max_mse:
        max_mse = mse
        max_idx = i
    if mse < min_mse:
        min_mse = mse
        min_idx = i

avg_bpp = avg_bpp / count
avg_mse = avg_mse / count
avg_l1 = avg_l1 / count
avg_msssim = avg_msssim / count
rms = np.sqrt(avg_mse)
psnr = -20*np.log10(rms)
print('min  MSE = ', min_mse)
print('max MSE = ', max_mse)
print('average MSE: ', avg_mse,',  ', avg_mse*255**2)
print('average RMS: ', rms,', ', rms*255)
print('average PSNR: ', psnr)
print('average MS-SSIM: ', avg_msssim)
print('average  bpp = ', avg_bpp)


plot_best_and_worst = True
if plot_best_and_worst:
    img1_minDist = Image.open(stereo1_path_list[min_idx])
    img2_minDist = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[min_idx]))
    img2_minDist = Image.open(img2_minDist)
    in1 = transform(img1_minDist)
    in2 = transform(img2_minDist)
    shape = in1.size()
    in1 = in1[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    in2 = in2[:, :M * (shape[1] // M), :M * (shape[2] // M)]
    img1_minDist = torch.squeeze(in1).permute(1, 2, 0).cpu().detach().numpy()
    input1 = in1[None, ...].to(device)
    input2 = in2[None, ...].to(device)
    # Encoded images:
    _, mse, final_im1_recon, _ = model(input1, input2)
    img1_minDist_rec = torch.squeeze(final_im1_recon).permute(1, 2, 0).cpu().detach().numpy()


    img1_maxDist = Image.open(stereo1_path_list[max_idx])
    img2_maxDist = Image.open(os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[max_idx])))
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
    _, mse, final_im1_recon, _ = model(input1, input2)
    img1_maxDist_rec = torch.squeeze(final_im1_recon).permute(1, 2, 0).cpu().detach().numpy()
    #img1_maxDist_rec = img1_maxDist
    ##
    print('the worst image is',stereo1_path_list[max_idx])


    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Best Hamming distance,')
    ax1.imshow(img1_minDist)
    ax2.imshow(img1_minDist_rec)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Worst Hamming distance,')
    ax1.imshow(img1_maxDist)
    ax2.imshow(img1_maxDist_rec)
    fig.tight_layout()

    plt.show()

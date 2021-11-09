import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import PIL
import glob
import numpy as np

import gzip
import pytorch_msssim
from models.PASSRnet import PASSRnet


#/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/passrnet/model_best_weights.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PASSRnet(1)

checkpoint = torch.load(pretrained_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

net = model.to(device)
net.eval()


path_to_rec = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/try-GPNN/reconstructed'
stereo_recon_path_list = glob.glob(os.path.join(path_to_rec, '*.png'))


transform = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])
#transform = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])
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

#file_msssim = open('/home/access/dev/DSIN/sharons code/ToSend/images/5.values_list/KITTI_stereo/twoStepsCompression/msssim_list_KITTI_stereo_target_0.031bpp_twoStepsNet.txt',"a")
#file_bpp = open('/home/access/dev/DSIN/sharons code/ToSend/images/5.values_list/KITTI_stereo/twoStepsCompression/bpp_list_KITTI_stereo_target_0.031bpp_twoStepNet.txt',"a")

for i in range(len(stereo_recon_path_list)):
    im_rec = Image.open(stereo_recon_path_list[i])
    im_original = Image.open(stereo_recon_path_list[i].replace('reconstructed', 'original'))
    im_si = Image.open(stereo_recon_path_list[i].replace('reconstructed', 'SI'))

    im_rec = transform(im_rec)
    im_original = transform(im_original)
    im_si = transform(im_si)
    ##
    im_rec = im_rec[None, ...].to(device)
    im_original = im_original[None, ...].to(device)
    im_si = im_si[None, ...].to(device)

    # Encoded images:

    sr_recon_left = model(im_rec, im_si, is_training=True)

    numpy_input_image = torch.squeeze(im_original).permute(1, 2, 0).cpu().detach().numpy()
    tensor_output_image = torch.squeeze(sr_recon_left).permute(1, 2, 0)
    numpy_output_image = tensor_output_image.cpu().detach().numpy()
    l1 = np.mean(np.abs(numpy_input_image - numpy_output_image))
    mse = np.mean(np.square(numpy_input_image - numpy_output_image))  # * 255**2   #mse_loss.item()/2
    psnr = -20*np.log10(np.sqrt(mse))
    msssim = 1#ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0, size_average=True, win_size=11) ## should be 11 for full size, 7 for small
    #msssim = pytorch_msssim.ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0)
    if not msssim == msssim:
        print('1')



    #file_msssim.write(str(msssim.item()) + '\n')
    #file_bpp.write(str(bpp) + '\n')

    print(psnr, msssim)
    avg_msssim += msssim
    avg_l1 = avg_l1 + l1
    avg_mse = avg_mse + mse
    avg_psnr += psnr
    count = count + 1
    if mse > max_mse:
        max_mse = mse
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
    im_rec = Image.open(stereo_recon_path_list[min_idx])
    im_original = Image.open(stereo_recon_path_list[min_idx].replace('reconstructed', 'original'))
    im_si = Image.open(stereo_recon_path_list[min_idx].replace('reconstructed', 'SI'))

    im_rec = transform(im_rec)
    im_original = transform(im_original)
    im_si = transform(im_si)
    ##
    im_rec = im_rec[None, ...].to(device)
    im_original = im_original[None, ...].to(device)
    im_si = im_si[None, ...].to(device)
    # Encoded images:
    sr_recon_left = model(im_rec, im_si, is_training=True)
    img1_minDist_rec = torch.squeeze(sr_recon_left).permute(1, 2, 0).cpu().detach().numpy()


    im_rec = Image.open(stereo_recon_path_list[max_idx])
    im_original = Image.open(stereo_recon_path_list[max_idx].replace('reconstructed', 'original'))
    im_si = Image.open(stereo_recon_path_list[max_idx].replace('reconstructed', 'SI'))

    im_rec = transform(im_rec)
    im_original = transform(im_original)
    im_si = transform(im_si)
    ##
    im_rec = im_rec[None, ...].to(device)
    im_original = im_original[None, ...].to(device)
    im_si = im_si[None, ...].to(device)
    # Encoded images:
    sr_recon_left = model(im_rec, im_si, is_training=True)
    img1_maxDist_rec = torch.squeeze(sr_recon_left).permute(1, 2, 0).cpu().detach().numpy()
    #img1_maxDist_rec = img1_maxDist
    ##
    print('the worst image is',stereo_recon_path_list[max_idx])


    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Best MSE reconstruction,')
    ax1.imshow(Image.open(stereo_recon_path_list[min_idx].replace('reconstructed', 'original')))
    ax2.imshow(img1_minDist_rec)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Worst MSE reconstruction,')
    ax1.imshow(Image.open(stereo_recon_path_list[max_idx].replace('reconstructed', 'original')))
    ax2.imshow(img1_maxDist_rec)
    fig.tight_layout()

    plt.show()

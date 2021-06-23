
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import numpy as np
# from model import *
from model_new import *
from model_small import ImageCompressor_small


#pretrained_model_path ='/home/access/dev/iclr_17_compression/checkpoints_new/small image factor2/full-loss _ from pretrained/iter_240.pth.tar'
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/diff_img2_N=32/rec/iter_853.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = ImageCompressor_new()
#model = ImageCompressor_new(out_channel_N=16)
model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_small()
global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()


#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
#stereo1_dir = '/home/access/dev/data_sets/kitti/upsample - try/original'
# smaller dataset:
#stereo1_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_02'

# diff images:
stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/diff_image_2'


stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))
# transforms to fit model size
data_transforms = transforms.Compose([transforms.Resize((384, 1248),interpolation=3), transforms.ToTensor()])
#data_transforms = transforms.Compose([transforms.Resize((192, 624), interpolation=3), transforms.ToTensor()])
#data_transforms = transforms.Compose([transforms.Resize((96, 320), interpolation=3), transforms.ToTensor()])
#data_transforms = transforms.Compose([transforms.ToTensor()])


avg_mse = 0
avg_l1 = 0
min_mse = 1000
max_mse = 0
min_idx = 0
max_idx = 0
count = 0
for i in range(len(stereo1_path_list)):
    # Read Image File
    image1 = Image.open(stereo1_path_list[i])
    # Run image through the model
    tensor_image1 = data_transforms(image1)
    input1 = tensor_image1[None, ...].to(device)
    clipped_recon_image, _, _ = net(input1)

    # Calc MSE
    numpy_input_image = tensor_image1.permute(1, 2, 0).detach().numpy()
    tensor_output_image = torch.squeeze(clipped_recon_image).permute(1, 2, 0)
    numpy_output_image = tensor_output_image.cpu().detach().numpy()
    # clip values larger than 1:
    numpy_output_image[numpy_output_image > 1] = 1
    l1 = np.mean(np.abs(numpy_input_image-numpy_output_image))
    mse = np.mean(np.square(numpy_input_image-numpy_output_image)) #* 255**2

    if mse > max_mse:
        max_mse = mse
        max_idx = i
    if mse < min_mse:
        min_mse = mse
        min_idx = i
    print(mse, ", ", l1)
    avg_l1 = avg_l1 + l1
    avg_mse = avg_mse + mse
    count = count+1

avg_mse = avg_mse / count
avg_l1 = avg_l1 / count
rms = np.sqrt(avg_mse)
psnr = -20*np.log10(rms)
print('min  MSE = ', min_mse)
print('max MSE = ', max_mse)
print('average MSE: ', avg_mse,',  ', avg_mse*255**2)
print('average RMS: ', rms,', ', rms*255)
print('average PSNR: ', psnr)
print('avg L1: ', avg_l1*255)



plot_best_and_worst = True
if plot_best_and_worst:

    img1_minDist_input = Image.open(stereo1_path_list[min_idx])
    tensor_image1 = data_transforms(img1_minDist_input)
    input1 = tensor_image1[None, ...].to(device)
    clipped_recon_image, _, _ = net(input1)
    img1_minDist_input = tensor_image1.permute(1, 2, 0).detach().numpy()
    #img1_minDist_input = img1_minDist_input * 0.5 + 0.5

    tensor_output_image = torch.squeeze(clipped_recon_image).permute(1, 2, 0)
    img1_minDist_output = tensor_output_image.cpu().detach().numpy()
    #img1_minDist_output = img1_minDist_output*0.5 + 0.5
    img1_minDist_output[img1_minDist_output > 1] = 1

    img1_maxDist_input = Image.open(stereo1_path_list[max_idx])
    tensor_image1 = data_transforms(img1_maxDist_input)
    input1 = tensor_image1[None, ...].to(device)
    clipped_recon_image, _, _ = net(input1)
    img1_maxDist_input = tensor_image1.permute(1, 2, 0).detach().numpy()
    tensor_output_image = torch.squeeze(clipped_recon_image).permute(1, 2, 0)
    img1_maxDist_output = tensor_output_image.cpu().detach().numpy()
    img1_maxDist_output[img1_maxDist_output > 1] = 1

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Best MSE Reconstruction, RMS = '+str(format(np.sqrt(min_mse)*255,".2f")))
    ax1.imshow(img1_minDist_input)
    ax2.imshow(img1_minDist_output)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Worst MSE Reconstruction, RMS = '+str(format(np.sqrt(max_mse)*255,".2f")))
    ax1.imshow(img1_maxDist_input)
    ax2.imshow(img1_maxDist_output)
    fig.tight_layout()

    plt.show()
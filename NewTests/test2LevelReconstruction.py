
import torch.nn.functional as F
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageChops
import glob
import numpy as np
from model_new import *
from model import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the small images AE model
model_orig_weights = '/home/access/dev/iclr_17_compression/checkpoints_new/new loss - L2 before binarize/rec+hamm/iter_3.pth.tar'
#model = ImageCompressor_new()
model_orig = ImageCompressor_new(out_channel_N=256)
global_step_ignore = load_model(model_orig, model_orig_weights)
model_orig = model_orig.to(device)
model_orig.eval()

# Load the small images AE model
model_diff_weights = '/home/access/dev/iclr_17_compression/checkpoints/iter_117600.pth.tar'
#model_diff = ImageCompressor_new()
#model_diff = ImageCompressor_new(out_channel_N=32)
model_diff = ImageCompressor()
global_step_ignore = load_model(model_diff, model_diff_weights)
model_diff = model_diff.to(device)
model_diff.eval()

# Define transform for small(trained model) and original image size.
tsfm_original = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC)])
tsfm_original_tensor = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC), transforms.ToTensor()])
tsfm_tensor = transforms.Compose([transforms.ToTensor()])

#path = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_2'
path = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_02'
files = os.listdir(path)
avg_psnr = 0
for i in range(len(files)):

    file_name = os.path.join(path, files[i])
    image = Image.open(file_name)#.convert('RGB')

    # Get rec image from model_orig:
    img_input = tsfm_original_tensor(image)[None, ...].to(device)
    clipped_recon_image, z_cam1, _ = model_orig(img_input)
    img_original_recon = torch.squeeze(clipped_recon_image).permute(1, 2, 0).cpu().detach().numpy()
    # Get diff image from model_diff:
        ## Calc diff
    img_original_np = np.array(tsfm_original(image))
    diff = np.clip((127 + (img_original_np - img_original_recon * 255)), 0, 255).astype(np.uint8)
    diff_pil = Image.fromarray(diff)
        ## send through model_diff
    diff_input = tsfm_tensor(diff_pil)[None, ...].to(device)
    clipped_recon_image, z_cam1, _ = model_diff(diff_input)
    diff_recon = torch.squeeze(clipped_recon_image).permute(1, 2, 0).cpu().detach().numpy()

    # Combine two image to get final output:
    final_rec = (img_original_recon + diff_recon - 127/255)

    mse = np.mean(np.square(final_rec - img_original_np/255))
    rms = np.sqrt(mse)
    psnr = -20 * np.log10(rms)
    avg_psnr += psnr
    print(psnr)
avg_psnr = avg_psnr/len(files)

print('avg psnr = ', avg_psnr)

print('done')
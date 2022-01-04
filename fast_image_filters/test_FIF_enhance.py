import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset_FIF_enhance
import time
import torchvision
from losses import *
import numpy as np
import pytorch_msssim

from fast_image_filters.FIF_enhance_net import FIF_enhance
from fast_image_filters.final_enhance_net import finalEnhanceNet



path_to_reconstructed_images = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/try-GPNN/reconstructed'


weights_path = '/home/access/dev/iclr_17_compression/fast_image_filters/enhance_weights/model_best_weights.pth'

################ Data transforms ################
tsfm = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])


training_data = StereoDataset_FIF_enhance(path_to_reconstructed_images, tsfm, randomCrop=False)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load model:
#model = FIF_enhance()
model = finalEnhanceNet()
model = model.to(device)

checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


criterion_mse = nn.MSELoss()
criterion_L1 = nn.L1Loss()
# monitor training loss
train_loss = 0.0
# Training
epoch_start_time = time.time()
avg_psnr = 0.0
avg_msssim = 0.0
for batch, data in enumerate(train_dataloader):
    # Get stereo pair
    im_si, im_LR, im_HR = data
    im_si = im_si.to(device)
    im_LR = im_LR.to(device)
    im_HR = im_HR.to(device)

    recon = model(torch.cat((im_LR, im_si), 1))

    numpy_input_image = torch.squeeze(im_HR).cpu().permute(1, 2, 0).detach().numpy()
    numpy_output_image = torch.squeeze(recon).permute(1, 2, 0).cpu().detach().numpy()
    l1 = np.mean(np.abs(numpy_input_image - numpy_output_image))
    mse = np.mean(np.square(numpy_input_image - numpy_output_image))  # * 255**2   #mse_loss.item()/2
    psnr = -20 * np.log10(np.sqrt(mse))
    # msssim = ms_ssim(final_im1_recon.cpu().detach(), input1.cpu(), data_range=1.0, size_average=True, win_size=11) ## should be 11 for full size, 7 for small
    msssim = pytorch_msssim.ms_ssim(recon, im_HR, data_range=1.0)
    avg_psnr += psnr
    avg_msssim += msssim.item()
    print(psnr, msssim)
    if not msssim == msssim:
        print('1')


avg_msssim = avg_msssim / len(train_dataloader)
avg_psnr = avg_psnr / len(train_dataloader)

print('avg ms-ssim = ', avg_msssim)
print('avg psnr = ', avg_psnr)


print("Done!")





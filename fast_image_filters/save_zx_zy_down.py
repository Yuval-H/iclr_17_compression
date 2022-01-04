import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import PIL
import glob
from model import *
from models.temp import Cheng2020Attention



pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/4 bit - verify/model_bestVal_loss_msssim.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Cheng2020Attention()

checkpoint = torch.load(pretrained_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

net = model.to(device)
net.eval()

stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview/testing/image_2'
stereo2_dir = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview/testing/image_2'

# smaller dataset:
#stereo1_dir = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_2'

list1 = glob.glob(os.path.join(stereo1_dir, '*11.png'))
list2 = glob.glob(os.path.join(stereo2_dir, '*11.png'))
stereo1_path_list = list1 + list2
#stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*.png'))


transform = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])

rec_path_Zx = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/try_warping/rec_zx_down/'
rec_path_Zy = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/try_warping/rec_zy_down/'

for i in range(len(stereo1_path_list)):
    img_stereo1 = Image.open(stereo1_path_list[i])
    img_stereo2_name = stereo1_path_list[i].replace('image_2', 'image_3')
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

    # get Zx_down reconstruction:
    mse_loss, mse_on_full, final_zx_recon, z1_down = model(input1, input2)
    numpy_output_image = torch.squeeze(final_zx_recon).permute(1, 2, 0).cpu().detach().numpy()

    rec_img = Image.fromarray((numpy_output_image * 255).astype(np.uint8))
    rec_img.save(rec_path_Zx + stereo1_path_list[i][-13:])

    # get Zy_down reconstruction:
    mse_loss, mse_on_full, final_zx_recon, z1_down = model(input2, input1)
    numpy_output_image = torch.squeeze(final_zx_recon).permute(1, 2, 0).cpu().detach().numpy()

    rec_img = Image.fromarray((numpy_output_image * 255).astype(np.uint8))
    rec_img.save(rec_path_Zy + stereo1_path_list[i][-13:])



import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import numpy as np
from model_new import *
from model import *


#pretrained_model_path ='/home/access/dev/iclr_17_compression/checkpoints/iter_471527.pth.tar'
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/not-binary-try/rec + distance/iter_1300.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_new()
model = ImageCompressor()
global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()


#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
# smaller dataset:
stereo1_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_02'

stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))

# transforms to fit model size
#data_transforms = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC), transforms.ToTensor()])
data_transforms = transforms.Compose([transforms.Resize((96, 320), interpolation=3), transforms.ToTensor()])


img_n = 0
scenario_number = os.path.basename(stereo1_path_list[img_n])[:-6]
image1 = Image.open(stereo1_path_list[img_n])
avg_hamm_dist = 0
min_d = 1
max_d = 0
count = 0
for i in range(len(stereo1_path_list)):
    curr_scenario = os.path.basename(stereo1_path_list[i])[:-6]
    if curr_scenario == scenario_number:  # check distances for images from different scenarios
        continue

    image2 = Image.open(stereo1_path_list[i])
    tensor_image = data_transforms(image1)
    input1 = tensor_image[None, ...].to(device)
    tensor_image = data_transforms(image2)
    input2 = tensor_image[None, ...].to(device)
    # Encode images:
    outputs_cam1, encoded, _ = net(input1)
    outputs_cam1, encoded2, _ = net(input2)
    e1 = torch.squeeze(encoded.cpu()).detach().numpy().flatten()
    e2 = torch.squeeze(encoded2.cpu()).detach().numpy().flatten()
    hamm_dist = (e1 != e2).sum()
    nbits = e1.size
    hamm_dist_normalized = hamm_dist / nbits
    if hamm_dist_normalized > max_d:
        max_d = hamm_dist_normalized
    if hamm_dist_normalized < min_d:
        min_d = hamm_dist_normalized
    print(hamm_dist_normalized)
    avg_hamm_dist = avg_hamm_dist + hamm_dist_normalized
    count = count+1

avg_hamm_dist = avg_hamm_dist / count
print('average hamming distance: ', avg_hamm_dist)
print('min dist = ', min_d)
print('max dist = ', max_d)

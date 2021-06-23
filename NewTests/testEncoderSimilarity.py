import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import numpy as np
from model_new import *
from model_small import ImageCompressor_small


#pretrained_model_path ='/home/access/dev/iclr_17_compression/checkpoints/iter_471527.pth.tar'
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new loss - L1 before binarize/rec+hamm/iter_24.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_new()
#model = ImageCompressor_small()
global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()




stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
stereo2_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_3'

# smaller dataset:
#stereo1_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
#stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_03'

stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))

avg_hamm_dist = 0
min = 1
max = 0
min_idx = 0
max_idx = 0

#transform = transforms.Compose([transforms.Resize((384, 1216),interpolation=3), transforms.ToTensor()])
transform = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((192, 624),interpolation=3), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((96, 320), interpolation=3), transforms.ToTensor()])

for i in range(len(stereo1_path_list)):
    img_stereo1 = Image.open(stereo1_path_list[i])
    img_stereo2_name = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[i]))
    img_stereo2 = Image.open(img_stereo2_name)
    img_stereo1 = transform(img_stereo1)
    img_stereo2 = transform(img_stereo2)
    input1 = img_stereo1[None, ...].to(device)
    input2 = img_stereo2[None, ...].to(device)

    # Encoded images:
    outputs_cam1, encoded, _ = net(input1)
    outputs_cam2, encoded2, _ = net(input2)

    e1 = torch.squeeze(encoded.cpu()).detach().numpy().flatten()
    e2 = torch.squeeze(encoded2.cpu()).detach().numpy().flatten()
    hamm_dist = (e1 != e2).sum()
    nbits = e1.size
    hamm_dist_normalized = hamm_dist / nbits
    if hamm_dist_normalized > max:
        max = hamm_dist_normalized
        max_idx = i
    if hamm_dist_normalized < min:
        min = hamm_dist_normalized
        min_idx = i
    print(hamm_dist_normalized)
    avg_hamm_dist = avg_hamm_dist + hamm_dist_normalized
avg_hamm_dist = avg_hamm_dist/len(stereo1_path_list)
print('average hamming distance: ', avg_hamm_dist)
print('min dist = ', min)
print('max dist = ',max)


plot_best_and_worst = True
if plot_best_and_worst:
    img1_minDist = Image.open(stereo1_path_list[min_idx])
    img2_minDist = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[min_idx]))
    img2_minDist = Image.open(img2_minDist)

    img1_maxDist = Image.open(stereo1_path_list[max_idx])
    img2_maxDist = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[max_idx]))
    img2_maxDist = Image.open(img2_maxDist)


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Best Hamming distance,  ' + str(format(min*100, ".4f")) + '%')
    ax1.imshow(img1_minDist)
    ax2.imshow(img2_minDist)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Worst Hamming distance,  ' + str(format(max*100, ".2f")) + '%')
    ax1.imshow(img1_maxDist)
    ax2.imshow(img2_maxDist)
    fig.tight_layout()

    plt.show()


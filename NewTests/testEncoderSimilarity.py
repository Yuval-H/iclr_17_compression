import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import numpy as np
from model_new import *
from model import *
from model_small import ImageCompressor_small
from utils.Conditional_Entropy import compute_conditional_entropy


#pretrained_model_path ='/home/access/dev/iclr_17_compression/checkpoints/iter_471527.pth.tar'
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/N=1024/rec+hamm/iter_1.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImageCompressor_new(out_channel_N=1024)
#model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_new(out_channel_N=512)
#model = ImageCompressor_new()
#model = ImageCompressor_small()
#model = ImageCompressor()
global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()




#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
#stereo2_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_3'

# smaller dataset:
stereo1_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
stereo2_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_03'

stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))

avg_hamm_dist = 0
min = 1
max = 0
min_idx = 0
max_idx = 0

#transform = transforms.Compose([transforms.Resize((384, 1216),interpolation=3), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((192, 624),interpolation=3), transforms.ToTensor()])
#transform = transforms.Compose([transforms.Resize((96, 320), interpolation=3), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])

##z1_avg = np.array([0])[None,:]
##z2_avg = np.array([0])[None,:]
c_max = 0
c_min = 0
uncertainty_coefficient_avg = 0
mean_conditional_entropy_avg = 0

for i in range(len(stereo1_path_list)):
    img_stereo1 = Image.open(stereo1_path_list[i])
    img_stereo2_name = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[i]))
    img_stereo2 = Image.open(img_stereo2_name)
    img_stereo1 = transform(img_stereo1)
    img_stereo2 = transform(img_stereo2)
    # cut image H*W to be a multiple of 16
    shape = img_stereo1.size()
    img_stereo1 = img_stereo1[:, :16 * (shape[1] // 16), :16 * (shape[2] // 16)]
    img_stereo2 = img_stereo2[:, :16 * (shape[1] // 16), :16 * (shape[2] // 16)]
    ##
    input1 = img_stereo1[None, ...].to(device)
    input2 = img_stereo2[None, ...].to(device)

    ######### Temp patch:
    '''
    i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(128, 128))
    if i == 128:
        i = 127
    if j == 128:
        j = 127
    input1 = img_stereo1[:, i:i + h, j:j + w]
    input2 = img_stereo1[:, i:i + h, j+1:j + w +1]
    input1 = input1[None, ...].to(device)
    input2 = input2[None, ...].to(device)
    '''

    # Use center crop, shifted 33 pixel ~ vertical alignment
    #'''
    input1 = input1[:, :, :, 33:]
    input2 = input2[:, :, :, :-33]
    # cut image H*W to be a multiple of 16
    shape = input1.size()
    input1 = input1[:, :, :16 * (shape[2] // 16), :16 * (shape[3] // 16)]
    input2 = input2[:, :, :16 * (shape[2] // 16), :16 * (shape[3] // 16)]
    input1 = input1.to(device)
    input2 = input2.to(device)
    #'''
    ######### End Temp patch

    # Encoded images:
    outputs_cam1, encoded, _ = net(input1)
    outputs_cam2, encoded2, _ = net(input2)

    '''
    c1 = torch.squeeze(encoded.cpu()).detach().numpy()
    c2 = torch.squeeze(encoded2.cpu()).detach().numpy()
    diff = c1 - c2
    plt.hist(diff.flatten(), np.arange(diff.min(), diff.max()+1))
    '''

    e1 = torch.squeeze(encoded.cpu()).detach().numpy().flatten()
    e2 = torch.squeeze(encoded2.cpu()).detach().numpy().flatten()
    uncertainty_coefficient, mean_conditional_entropy = compute_conditional_entropy(e1, e2)
    uncertainty_coefficient_avg += uncertainty_coefficient
    mean_conditional_entropy_avg += mean_conditional_entropy
    #print(uncertainty_coefficient)
    max_curr = np.max((e1.max(), e2.max()))
    if max_curr > c_max:
        c_max = max_curr
    min_curr = np.min((e1.min(), e2.min()))
    if min_curr < c_min:
        c_min = min_curr
    ##z1_avg = np.concatenate((z1_avg, e1[None,:]), axis=1)
    ##z2_avg = np.concatenate((z2_avg, e2[None,:]), axis=1)
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
uncertainty_coefficient_avg = uncertainty_coefficient_avg/len(stereo1_path_list)
mean_conditional_entropy_avg = mean_conditional_entropy_avg/len(stereo1_path_list)

print('average uncertainty coefficient:', uncertainty_coefficient_avg)
print('average mean-conditional-entropy:', mean_conditional_entropy)
print('average hamming distance: ', avg_hamm_dist)
print('min dist = ', min)
print('max dist = ', max)


plot_best_and_worst = True
if plot_best_and_worst:
    img1_minDist = Image.open(stereo1_path_list[min_idx])
    img2_minDist = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[min_idx]))
    img2_minDist = Image.open(img2_minDist)

    img1_maxDist = Image.open(stereo1_path_list[max_idx])
    img2_maxDist = os.path.join(stereo2_dir, os.path.basename(stereo1_path_list[max_idx]))
    img2_maxDist = Image.open(img2_maxDist)
    print('the worst image is',stereo1_path_list[max_idx])


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


import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import numpy as np
from model_new import *
from model import *


#def plot_code_hist(code1):




# pretrained_model_path ='/home/access/dev/iclr_17_compression/checkpoints/iter_471527.pth.tar'
pretrained_model_path = '/home/access/dev/iclr_17_compression/checkpoints_new/Mnually shift networks/33 pixel stereo shift/iter_0_111_05.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImageCompressor_new(out_channel_N=1024)
#model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_new(out_channel_N=128)
#model = ImageCompressor_new()
#model = ImageCompressor()
global_step_ignore = load_model(model, pretrained_model_path)
net = model.to(device)
net.eval()

#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
stereo1_dir = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
#stereo1_dir = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/diff_image_2'

stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))

# transforms to fit model size
#data_transforms = transforms.Compose([transforms.Resize((384, 1216)), transforms.ToTensor()])
#data_transforms = transforms.Compose([transforms.Resize((384, 1248)), transforms.ToTensor()])

## To test on original images (no resizing)
data_transforms = transforms.Compose([transforms.CenterCrop((368, 1216)), transforms.ToTensor()])

p0_avg = 0
p1_avg = 0
for i in range(len(stereo1_path_list)):

    image = Image.open(stereo1_path_list[i])
    tensor_image = data_transforms(image)
    # cut image H*W to be a multiple of 16
    shape = tensor_image.size()
    tensor_image = tensor_image[:, :16 * (shape[1] // 16), :16 * (shape[2] // 16)]
    input1 = tensor_image[None, ...].to(device)
    # Encode image:
    outputs_cam, encoded, _ = net(input1)
    code = torch.squeeze(encoded.cpu()).detach().numpy().flatten()

    if i == 0:
        avg_code = encoded.cpu().detach().numpy()
    else:
        avg_code += encoded.cpu().detach().numpy()
    # Calc p0,p1 and code distribution
    p1 = np.mean(code == 1)
    p1_avg += p1
    p0 = np.mean(code == 0)
    p0_avg += p0

avg_code = np.round(avg_code / len(stereo1_path_list))

# init unchained bit - binary array of 0/1 = changed/unchanged
test_unchained_bits = np.ones(avg_code.shape)

calc_var = True
if calc_var:
    for i in range(len(stereo1_path_list)):
        image = Image.open(stereo1_path_list[i])
        tensor_image = data_transforms(image)
        input1 = tensor_image[None, ...].to(device)
        # Encode image:
        outputs_cam, encoded, _ = net(input1)

        # Test Unchanged bits:
        test_unchained_bits = test_unchained_bits * (encoded.cpu().detach().numpy() == avg_code).astype(np.uint8)
        # Calc Var
        if i == 0:
            var_code = np.abs(encoded.cpu().detach().numpy()-avg_code)
        else:
            var_code += np.abs(encoded.cpu().detach().numpy()-avg_code)

    var_code = var_code / len(stereo1_path_list)
    var_hist = np.mean(var_code[0,:,:,:], axis=(1, 2))
    plt.plot(var_hist, 'bo')
    plt.title('avg-var-histogram of 16 z channels')
    plt.xlabel('depth Z channel')
    plt.ylabel('avg-var over ch')
    plt.show()

hist = np.mean(avg_code[0,:,:,:], axis=(1, 2))
#hist = np.mean(avg_code.reshape(-1, 4608), axis=1)
plt.plot(hist, 'bo')
plt.title('avg-histogram of 16 z channels')
plt.xlabel('depth Z channel')
plt.ylabel('avg over ch')
plt.show()

p0_avg = p0_avg / len(stereo1_path_list)
p1_avg = p1_avg / len(stereo1_path_list)
print('Average P(0)= ', format(p0_avg, ".3f"), '\nAverage P(1)=', format(p1_avg, ".3f"))
unchanged_precentage = np.sum(test_unchained_bits)/test_unchained_bits.size
print('% of unchanged bits = ', format(unchanged_precentage, ".3f"))





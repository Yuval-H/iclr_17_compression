

from torchvision import transforms
#import matplotlib.pyplot as plt
from PIL import Image, ImageChops
#import numpy as np
from model_new import *


# Load the small images AE model
model_weights = '/home/access/dev/iclr_17_compression/checkpoints_new/new loss - L2 before binarize/rec+hamm/iter_3.pth.tar'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = ImageCompressor_new()
model = ImageCompressor_new(out_channel_N=256)
global_step_ignore = load_model(model, model_weights)
net = model.to(device)
net.eval()

# Define transform for small(trained model) and original image size.
tsfm_original = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC)])
tsfm_original_tensor = transforms.Compose([transforms.Resize((384, 1248), interpolation=Image.BICUBIC), transforms.ToTensor()])


path = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_2'
save_path = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/diff_image_2'
files = os.listdir(path)

for i in range(len(files)):

    file_name = os.path.join(path, files[i])
    image = Image.open(file_name)
    img_original_np = np.array(tsfm_original(image))
    img_input = tsfm_original_tensor(image)[None, ...].to(device)
    # Get network reconstructed image output:
    clipped_recon_image, _, _ = net(img_input)
    # Up-sample to original size:
    img_original_recon = torch.squeeze(clipped_recon_image)
    img_original_recon = img_original_recon.permute(1, 2, 0).cpu().detach().numpy()

    diff = np.clip((127 + (img_original_np - img_original_recon*255)), 0, 255).astype(np.uint8)
    diff_pil = Image.fromarray(diff)


    diff_pil.save(os.path.join(save_path, files[i]))
print('done')
import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset, StereoPlusDataset
import time
import torchvision
from losses import *
from model_new import *
from model import *
from model_fc import *
from model_small import ImageCompressor_small
from models.temp import Cheng2020Attention

from compressai.zoo import cheng2020_attn



############## Train parameters ##############
#train_folder1 = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
#train_folder2 = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_03'
#train_folder1 = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/train'
#train_folder2 = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/train'

train_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_2'
train_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_3'
val_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
val_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_3'

batch_size = 1
lr_start = 1e-5
epoch_patience = 60
n_epochs = 25000
val_every = 25000
save_every = 2000
using_blank_loss = False
hammingLossOnBinaryZ = False
useStereoPlusDataSet = False
start_from_pretrained = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/using 22 nets on latent/try_L1/iter_11.pth.tar'
save_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/using 22 nets on latent/try_L1'

################ Data transforms ################
tsfm = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(),
#                                transforms.RandomVerticalFlip(), transforms.ToTensor()])


######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

training_data = StereoDataset(stereo1_dir=train_folder1, stereo2_dir=train_folder2, randomFlip=False, RandomCrop=False, transform=tsfm)
val_data = StereoDataset(stereo1_dir=val_folder1, stereo2_dir=val_folder2, transform=tsfm)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load model:
#net1 = cheng2020_attn(quality=4).to(device)
model = Cheng2020Attention()
if start_from_pretrained != '':
    global_step_ignore = load_model(model, start_from_pretrained)
model = model.to(device)
model.train()

freez2_base_autoencoder = False
if freez2_base_autoencoder:
    for param in model.g_a.parameters():
        param.requires_grad = False
    for param in model.g_s.parameters():
        param.requires_grad = False


# todo: check with & without
clipping_value = 5.0
torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)


optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epoch_patience, verbose=True)

# Epochs
best_loss = 10000
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    M = 32
    epoch_start_time = time.time()
    for batch, data in enumerate(train_dataloader):
        # Get stereo pair
        images_cam1, images_cam2 = data
        # Cut to be multiple of 32 (M)
        shape = images_cam1.size()
        images_cam1 = images_cam1[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
        images_cam2 = images_cam2[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
        images_cam1 = images_cam1.to(device)
        images_cam2 = images_cam2.to(device)

        optimizer.zero_grad()

        mse_1, mse_2, mse_z = model(images_cam1, images_cam2)

        #loss = mse_2    #only final rec loss
        loss = mse_1 + mse_2   #final and backbone rec loss
        #loss = mse_1 + mse_2 + 0.5*mse_z
        #loss = mse_2 + 0.5 * mse_z
        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* images_cam1.size(0)

    train_loss = train_loss / len(train_dataloader)
    # Note that step should be called after validate()
    scheduler.step(train_loss)
    if train_loss < best_loss:
        best_loss = train_loss
        save_model(model, 1, save_path) #save_model(model, epoch, save_path)#torch.save(model.state_dict(), '../model_best_weights.pth')
    elif epoch % save_every == 0:
        save_model(model, epoch, save_path)


    #Training
    #Validation
    if epoch % val_every == 0:
        # validate
        model.eval()
        val_loss = 0
        for batch, data in enumerate(val_dataloader):
            # Get stereo pair
            images_cam1, images_cam2 = data
            # Cut to be multiple of 32 (M)
            shape = images_cam1.size()
            images_cam1 = images_cam1[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
            images_cam2 = images_cam2[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
            images_cam1 = images_cam1.to(device)
            images_cam2 = images_cam2.to(device)

            # get model outputs
            mse_1, mse_2, mse_z = model(images_cam1, images_cam2)
            loss = mse_1 + mse_2 + 0.5 * mse_z
            val_loss += loss.item()  # * images_cam1.size(0)
        model.train()
        val_loss = val_loss / len(val_dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f}\tVal Loss: {:.6f}\tEpoch Time: {:.6f}'
              .format(epoch, train_loss,val_loss, time.time() - epoch_start_time), end="\r")
    else:
        print('Epoch: {} \tTraining Loss: {:.6f}\tEpoch Time: {:.6f}'.format(epoch, train_loss, time.time() - epoch_start_time))

torch.save(model.state_dict(), 'model_weights.pth')
print("Done!")





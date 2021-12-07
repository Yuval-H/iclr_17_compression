import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset, StereoPlusDataset, StereoDataset_new, StereoDataset_HoloPix50k
import time
import torchvision
from losses import *
from model_new import *
from models.temp import Cheng2020Attention
from models.temp_reg_0_0625 import Cheng2020Attention_reg_0_0625

import kornia

import pytorch_msssim

####

############## Train parameters ##############

stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview'
stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview'
#stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_2'
#stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_2'
#stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_16/image_2'
#stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_16/image_2'

#path_holoPix_left_train = '/home/access/dev/Holopix50k/train/left'
#path_holoPix_left_test = '/home/access/dev/Holopix50k/test/left'

batch_size = 6
lr_start = 1e-4
epoch_patience = 5

n_epochs = 25000
val_every = 1000
save_every = 2000
start_from_pretrained = ''
save_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/Regressive model'

################ Data transforms ################
tsfm = transforms.Compose([transforms.ToTensor()])
tsfm_val = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])


######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

training_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=True, randomFlip=True,
                                  RandomCrop=True, crop_352_1216=False, colorJitter=False, transform=tsfm)
val_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=False, transform=tsfm_val)

#training_data = StereoDataset_HoloPix50k(path_holoPix_left_train, RandomCrop=True, transform=tsfm)
#val_data = StereoDataset_HoloPix50k(path_holoPix_left_test, transform=tsfm_val)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Load base model   -  0.03125 bpp:
path_base_model = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/4 bit - verify/start_from_holopix/model_bestVal_loss_bestSoFar.pth'
model_0_031 = Cheng2020Attention()
checkpoint = torch.load(path_base_model)
model_0_031.load_state_dict(checkpoint['model_state_dict'])
model_0_031 = model_0_031.to(device)
model_0_031.eval()

# Load regressive model - adds another 0.03125 bpp
model = Cheng2020Attention_reg_0_0625()
model = model.to(device)
model.train()


optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epoch_patience, verbose=True)

epoch_start = 1
if start_from_pretrained != '':

    checkpoint = torch.load(start_from_pretrained)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #epoch_start = checkpoint['epoch']
    #loss = checkpoint['loss']

model.train()


# todo: check with & without
clipping_value = 5.0
torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)


# Epochs
best_loss = 10000
best_val_loss = 10000
lossL1 = nn.L1Loss()
lossMSE = nn.MSELoss()
for epoch in range(epoch_start, n_epochs + 1):
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

        # Get Z_2 and Z1_hat from 0.031_bpp_model
        _, _, im_rec_base, _ = model_0_031(images_cam1, images_cam2)

        # add regressive data  - using regressive model
        res_rec = model(images_cam1, images_cam2)

        final_rec = im_rec_base + res_rec

        #rec_loss = lossL1(images_cam1, final_rec)
        #res_zero_loss = 1/(1 + lossMSE(res_rec, torch.zeros_like(res_rec).to(device)))
        #loss_laplacian = lossL1(kornia.filters.laplacian(final_rec,3), kornia.filters.laplacian(images_cam1,3))
        #loss = 10*rec_loss + res_zero_loss
        loss_msssim = 1 - pytorch_msssim.ms_ssim(final_rec, images_cam1, data_range=1.0)

        loss = loss_msssim#rec_loss + 5*loss_laplacian

        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* images_cam1.size(0)

    train_loss = train_loss / len(train_dataloader)
    # Note that step should be called after validate()
    scheduler.step(train_loss)
    if train_loss < best_loss:
        best_loss = train_loss

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
        }, save_path+'/model_best_weights.pth')
    elif epoch % save_every == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
        }, save_path+'/model_weights_epoch_' + str(epoch) + '.pth')

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

            # Get Z_2 and Z1_hat from 0.031_bpp_model
            #(z1_hat_1, z2), _, _, _ = model_0_031(images_cam1, images_cam2)
            # add regressive data  - using regressive model
            #rec_loss, _, _ = model(images_cam1, z2, z1_hat_1)

            # Get Z_2 and Z1_hat from 0.031_bpp_model
            _, _, im_rec_base, _ = model_0_031(images_cam1, images_cam2)

            # add regressive data  - using regressive model
            res_rec,_ = model(images_cam1, images_cam2)
            final_rec = im_rec_base + res_rec
            rec_loss = lossL1(images_cam1, final_rec)

            val_loss += rec_loss.item()  # * images_cam1.size(0)
        model.train()
        val_loss = val_loss / len(val_dataloader)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
            }, save_path + '/model_bestVal_loss.pth')
        print('Epoch: {} \tTraining Loss: {:.6f}\tVal Loss: {:.6f}\tEpoch Time: {:.6f}'
              .format(epoch, train_loss, val_loss, time.time() - epoch_start_time))#, end="\r")
    else:
        print('Epoch: {} \tTraining Loss: {:.6f}\tEpoch Time: {:.6f}'.format(epoch, train_loss, time.time() - epoch_start_time))

torch.save(model.state_dict(), 'model_weights.pth')
print("Done!")





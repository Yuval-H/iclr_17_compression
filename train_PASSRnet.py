import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset_passrNet
import time
import torchvision
from losses import *

from models.PASSRnet import PASSRnet



############## Train parameters ##############

path_to_reconstructed_images = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/try-GPNN/reconstructed'

batch_size = 1
lr_start = 1e-4
epoch_patience = 20
n_epochs = 25000
val_every = 25000
save_every = 2000
using_blank_loss = False
hammingLossOnBinaryZ = False
useStereoPlusDataSet = False
start_from_pretrained = ''
save_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/Sharons dataset/passrnet'

################ Data transforms ################
tsfm = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.CenterCrop((320, 640)), transforms.ToTensor()])
tsfm_val = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.ToTensor()])
#transforms.Resize((160, 160), interpolation=3)



######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

training_data = StereoDataset_passrNet(path_to_reconstructed_images, tsfm, randomCrop=True)
val_data = StereoDataset_passrNet(path_to_reconstructed_images, tsfm_val)


train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Load model:
model = PASSRnet(upscale_factor=1)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, weight_decay=1e-8, momentum=0.9)
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


# Epochs
best_loss = 10000
best_val_loss = 10000
for epoch in range(epoch_start, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    epoch_start_time = time.time()
    loss_l1 = nn.L1Loss()
    for batch, data in enumerate(train_dataloader):
        # Get stereo pair
        recon_left, im_left, im_right = data
        recon_left = recon_left.to(device)
        im_left = im_left.to(device)
        im_right = im_right.to(device)

        optimizer.zero_grad()

        #mse_1, mse_2, mse_z, img_recon = model(images_cam1, images_cam2)
        sr_recon_left = model(recon_left, im_right, is_training=True)

        #msssim = ms_ssim(images_cam1, img_recon, data_range=1.0, size_average=True, win_size=11) ## should be 11 for full size, 7 for small
        #msssim = pytorch_msssim.ms_ssim(images_cam1, img_recon, data_range=1.0)

        #if not msssim == msssim:
        #    print('nan value')

        loss = loss_l1(sr_recon_left, im_left)
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
        #save_model(model, 1, save_path) #save_model(model, epoch, save_path)
    elif epoch % save_every == 0:
        #save_model(model, epoch, save_path)
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

            # get model outputs
            _, mse_2, _, _ = model(images_cam1, images_cam2)
            loss = mse_2  # only final rec loss
            #loss = mse_1 + mse_2 + 0.5 * mse_z
            val_loss += loss.item()  # * images_cam1.size(0)
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





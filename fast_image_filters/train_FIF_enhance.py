import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset_FIF_enhance
import time
import torchvision
from losses import *
import pytorch_msssim

from fast_image_filters.FIF_enhance_net import FIF_enhance
from fast_image_filters.temp_fif_enhance import temp_FIF_enhance


############## Train parameters ##############

path_to_reconstructed_images = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/try-GPNN/reconstructed'

batch_size = 1
lr_start = 1e-4
epoch_patience = 5
n_epochs = 25000
val_every = 25000
save_every = 2000
using_blank_loss = False
hammingLossOnBinaryZ = False
useStereoPlusDataSet = False
start_from_pretrained = ''
save_path = '/home/access/dev/iclr_17_compression/fast_image_filters/enhance_weights'

################ Data transforms ################
tsfm = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.CenterCrop((320, 640)), transforms.ToTensor()])
tsfm_val = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.ToTensor()])
#transforms.Resize((160, 160), interpolation=3)



######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

training_data = StereoDataset_FIF_enhance(path_to_reconstructed_images, tsfm, randomCrop=True)
val_data = StereoDataset_FIF_enhance(path_to_reconstructed_images, tsfm, randomCrop=False)


train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Load model:
#model = FIF_enhance()
model = temp_FIF_enhance()
model = model.to(device)

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


# Epochs
best_loss = 10000
best_val_loss = 10000
#criterion = nn.L1Loss()
criterion_mse = nn.MSELoss()
criterion_L1 = nn.L1Loss()
for epoch in range(epoch_start, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0
    # Training
    epoch_start_time = time.time()
    for batch, data in enumerate(train_dataloader):
        # Get stereo pair
        #im_rec_im_si, im_rec, HR_left = data
        im_si, im_LR, im_HR = data
        im_si = im_si.to(device)
        im_LR = im_LR.to(device)
        im_HR = im_HR.to(device)

        optimizer.zero_grad()

        #recon = im_LR + model(torch.cat((im_LR, im_si),1))
        recon = model(torch.cat((im_LR, im_si), 1))
        ###SR_left = model(LR_left, HR_right, is_training=True)

        #msssim = pytorch_msssim.ms_ssim(images_cam1, img_recon, data_range=1.0)
        #if not msssim == msssim:
        #    print('nan value')

        #loss = criterion(SR_left, HR_left)

        ### loss_SR
        #loss = criterion_mse(recon, HR_left)
        loss = criterion_L1(recon, im_HR)
        #loss = 1 - pytorch_msssim.ms_ssim(recon.clamp(0., 1.), HR_left, data_range=1.0)

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
            LR_left, HR_right, HR_left = data
            LR_left = LR_left.to(device)
            HR_left = HR_left.to(device)
            HR_right = HR_right.to(device)

            # get model outputs
            SR_left = model(LR_left, HR_right, is_training=False)
            loss = criterion_L1(SR_left, HR_left)
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





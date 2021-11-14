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
#stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview'
#stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview'
stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview'
stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview'

batch_size = 1
lr_start = 1e-4
epoch_patience = 6
n_epochs = 25000
val_every = 1
save_every = 2000
using_blank_loss = False
hammingLossOnBinaryZ = False
useStereoPlusDataSet = False
start_from_pretrained = '/home/access/dev/weights-passr/model_best_weights1.pth'
save_path = '/home/access/dev/weights-passr'

################ Data transforms ################
tsfm = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.CenterCrop((320, 640)), transforms.ToTensor()])
tsfm_val = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.ToTensor()])
#transforms.Resize((160, 160), interpolation=3)



######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

training_data = StereoDataset_passrNet(stereo_dir_2012, stereo_dir_2015, tsfm, randomCrop=True)
val_data = StereoDataset_passrNet(stereo_dir_2012, stereo_dir_2015, tsfm, randomCrop=True, isTrainingData=False)


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
        LR_left, HR_right, HR_left = data
        b, c, h, w = LR_left.shape
        LR_left = LR_left.to(device)
        HR_left = HR_left.to(device)
        HR_right = HR_right.to(device)

        optimizer.zero_grad()

        SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
        (V_left_to_right, V_right_to_left) = model(LR_left, HR_right, is_training=True)
        ###SR_left = model(LR_left, HR_right, is_training=True)

        #msssim = pytorch_msssim.ms_ssim(images_cam1, img_recon, data_range=1.0)
        #if not msssim == msssim:
        #    print('nan value')

        #loss = criterion(SR_left, HR_left)

        ### loss_SR
        loss_SR = criterion_mse(SR_left, HR_left)

        ### loss_smoothness
        loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                 criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
        loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                 criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
        loss_smooth = loss_w + loss_h

        ### loss_cycle
        Identity = (torch.eye(w, w).repeat(b, h, 1, 1)).to(device)
        loss_cycle = criterion_L1(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3),
                                  Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                     criterion_L1(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3),
                                  Identity * V_right_to_left.permute(0, 2, 1, 3))

        ### loss_photometric
        HR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                    HR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        HR_right_warped = HR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                   LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        loss_photo = criterion_L1(LR_left * V_left_to_right, HR_right_warped * V_left_to_right) + \
                     criterion_L1(HR_right * V_right_to_left, LR_left_warped * V_right_to_left)

        ### losses
        loss = loss_SR + 0.005 * (loss_photo + loss_smooth + loss_cycle)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* images_cam1.size(0)

    train_loss = train_loss / len(train_dataloader)
    # Note that step should be called after validate()
    #scheduler.step(train_loss)
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





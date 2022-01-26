import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset, StereoPlusDataset, StereoDataset_new, StereoDataset_HoloPix50k
import time
import torchvision
from losses import *
from model_new import *
#from model import *
from models.bottleneck_Att import BottleneckAttention_modified
from models.temp_1bpp import Cheng2020Attention_1bpp



pretrained_model_path = '/media/access/SDB500GB/dev/iclr_17_compression/ckpoints_newest/checkpoints_new/new_net/Sharons dataset/1bpp net (0.125bpp)/model_bestVal_loss_fullMSSSIM.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_original = Cheng2020Attention_1bpp()

checkpoint = torch.load(pretrained_model_path)
model_original.load_state_dict(checkpoint['model_state_dict'])
model_original.to(device)
model_original.eval()




#val_folder1 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
#val_folder2 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_03'

stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview'
stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview'
#stereo_dir_2012 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_2/image_2'
#stereo_dir_2015 = '/media/access/SDB500GB/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_2/image_2'


#path_holoPix_left_train = '/home/access/dev/Holopix50k/train/left'
#path_holoPix_left_test = '/home/access/dev/Holopix50k/test/left'
#path_holoPix_left_train = '/home/yuvalh/holopix50k/DATA/Holopix50k/train/left'
#path_holoPix_left_test = '/home/yuvalh/holopix50k/DATA/Holopix50k/test/left'
test_model = False
batch_size = 1
lr_start = 1e-4
epoch_patience = 16
n_epochs = 25000
val_every = 25000
save_every = 2000

start_from_pretrained = ''
save_path = '/media/access/SDB500GB/dev/iclr_17_compression/ckpoints_newest/checkpoints_new/att_net/train_on_frezzed_E1/'

################ Data transforms ################
tsfm = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.CenterCrop((370, 740)),transforms.Resize((128, 256), interpolation=3), transforms.ToTensor()])
tsfm_val = transforms.Compose([transforms.CenterCrop((320, 1224)), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.CenterCrop((320, 320)), transforms.ToTensor()])
#tsfm_val = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.RandomCrop((370, 740)), transforms.Resize((128, 256), interpolation=3), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(),
#                                transforms.RandomVerticalFlip(), transforms.ToTensor()])


######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

#training_data = StereoDataset(stereo1_dir=train_folder1, stereo2_dir=train_folder2, randomFlip=False, RandomCrop=True,
#                              transform=tsfm, crop_352_1216=False)
#val_data = StereoDataset(stereo1_dir=val_folder1, stereo2_dir=val_folder2, transform=tsfm_val, RandomCrop=False, crop_352_1216=False)

training_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=True, randomFlip=False,
                                  RandomCrop=True, crop_352_1216=False, colorJitter=False, transform=tsfm)
val_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=False, transform=tsfm_val)

#training_data = StereoDataset_HoloPix50k(path_holoPix_left_train, RandomCrop=True, transform=tsfm)
#val_data = StereoDataset_HoloPix50k(path_holoPix_left_test, transform=tsfm_val)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Load model:
model = BottleneckAttention_modified(dim=128, dim_head=1024)
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

if test_model:
    model.eval()
else:
    model.train()

# todo: check with & without
#clipping_value = 5.0
#torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

lossL1 = nn.L1Loss()
# Epochs
best_loss = 10000
best_val_loss = 10000

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

        ###optimizer.zero_grad()

        #mse_1, mse_2, mse_z, img_recon = model(images_cam1, images_cam2)
        z1,z2 = model_original(images_cam1, images_cam2, returnZ1Z2=True)
        im1_from_im2 = model(z1, z2, images_cam2)

        loss = lossL1(images_cam1, im1_from_im2)
        #loss = 1 - pytorch_msssim.ms_ssim(im1_from_im2.clamp(0., 1.), images_cam1, data_range=1.0)

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
        }, save_path+'model_best_weights.pth')
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
            #_, mse_2, img_recon, _ = model(images_cam1, images_cam2)
            _, mse_2, img_recon, _ = model(images_cam1, images_cam2)#, masked_channels)
            #msssim = pytorch_msssim.ms_ssim(images_cam1, img_recon, data_range=1.0)
            #loss = 1 - msssim
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





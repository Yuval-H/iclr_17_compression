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
train_folder1 = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_02'
train_folder2 = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_32/image_03'
#train_folder1 = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/train'
#train_folder2 = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/train'

#train_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_2'
#train_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_3'
val_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
val_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_3'

contrast_folder = '/home/access/dev/data_sets/CLIC2021/professional_train_2020/train'
batch_size = 1
lr_start = 1e-4
epoch_patience = 100
n_epochs = 25000
val_every = 25000
save_every = 2000
using_blank_loss = False
hammingLossOnBinaryZ = False
useStereoPlusDataSet = False
start_from_pretrained = ''
save_path = '/home/access/dev/iclr_17_compression/checkpoints_new/new_net/rec+hamm'

################ Data transforms ################
#tsfm = transforms.Compose([transforms.Resize((384, 1216), interpolation=3), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.Resize((384, 1248), interpolation=3), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.Resize((192, 624), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.Resize((96, 320), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
tsfm = transforms.Compose([transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.RandomResizedCrop(256), transforms.RandomHorizontalFlip(),
#                                transforms.RandomVerticalFlip(), transforms.ToTensor()])


######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

if useStereoPlusDataSet:
    training_data = StereoPlusDataset(stereo1_dir=train_folder1, stereo2_dir=train_folder2,contrast_dir=contrast_folder,RandomCrop=False, transform=tsfm)
    val_data = StereoPlusDataset(stereo1_dir=val_folder1, stereo2_dir=val_folder2,contrast_dir=contrast_folder, transform=tsfm)
else:
    training_data = StereoDataset(stereo1_dir=train_folder1, stereo2_dir=train_folder2, randomFlip=False, RandomCrop=True, transform=tsfm)
    val_data = StereoDataset(stereo1_dir=val_folder1, stereo2_dir=val_folder2, transform=tsfm)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load model:
#net1 = cheng2020_attn(quality=4).to(device)

#model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_new(out_channel_N=512)
#model = ImageCompressor_new(out_channel_N=1024)
#model = ImageCompressor()
model = Cheng2020Attention()
if start_from_pretrained != '':
    global_step_ignore = load_model(model, start_from_pretrained)
model = model.to(device)
model.train()

# todo: check with & without
clipping_value = 5.0
torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

#criterion = MSE_and_Contrastive_loss(eps=0.01)
criterion = MSE_and_pairHamming_loss(eps=0.1)
#criterion = L1_and_pairHamming_loss(eps=1)
#criterion = MSE_and_blankContrastiveLoss(eps=1, margin=0.01)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epoch_patience, verbose=True)

# Epochs
best_loss = 10000
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    epoch_start_time = time.time()
    for batch, data in enumerate(train_dataloader):
        if useStereoPlusDataSet:
            images_cam1, images_cam2, image_contrast = data
            images_cam1 = images_cam1.to(device)
            images_cam2 = images_cam2.to(device)
            image_contrast = image_contrast.to(device)
        else:
            images_cam1, images_cam2 = data
            images_cam1 = images_cam1.to(device)
            images_cam2 = images_cam2.to(device)

        optimizer.zero_grad()

        ######### Temp patch:
        # Use 128*128 1 pixel shift
        '''
        i, j, h, w = transforms.RandomCrop.get_params(images_cam1, output_size=(128, 128))
        if i == 128:
            i = 127
        if j == 128:
            j = 127
        input1 = images_cam1[:, :,  i:i + h, j:j + w]
        input2 = images_cam1[:, :, i:i + h, j + 1:j + w + 1]
        images_cam1 = input1.to(device)
        images_cam2 = input2.to(device)
        if images_cam2.size(2) != 128 or images_cam2.size(3) != 128:
            print('something went wrong with cropping the images')
        '''
        # Use center crop, shifted 33 pixel ~ vertical alignment
        '''
        input1 = images_cam1[:, :, :, 33:]
        input2 = images_cam2[:, :, :, :-33]
        # cut image H*W to be a multiple of 16
        shape = input1.size()
        input1 = input1[:, :, :16 * (shape[2] // 16), :16 * (shape[3] // 16)]
        input2 = input2[:, :, :16 * (shape[2] // 16), :16 * (shape[3] // 16)]
        images_cam1 = input1.to(device)
        images_cam2 = input2.to(device)
        '''
        ######### End Temp patch
        #a = net1(images_cam1)
        using_new_net = True
        if using_new_net:
            mse_1, mse_2 = model(images_cam1, images_cam2)
        else:
            outputs_cam1, z_binary1, z_cam1 = model(images_cam1)
            outputs_cam2, z_binary2, z_cam2 = model(images_cam2)
            if hammingLossOnBinaryZ:
                z_cam1 = z_binary1
                z_cam2 = z_binary2

        if using_blank_loss:
            if epoch%2 == 0:
                blank_im = torch.ones_like(images_cam1)*torch.rand(1).to(device)
            else:
                blank_im = torch.rand_like(images_cam1).to(device)
            out_im, z__blank_binary, z_blank_im = model(blank_im)
            loss = criterion(outputs_cam1, z_cam1, outputs_cam2, z_cam2, images_cam1, images_cam2, z_blank_im)
        elif useStereoPlusDataSet:
            if epoch % 2 == 0:
                _, z_contrast, _ = model(image_contrast)
                loss = criterion(outputs_cam1, z_cam1, outputs_cam2, z_cam2, images_cam1, images_cam2, z_contrast)
            else:
                blank_im = torch.ones_like(images_cam1) * torch.rand(1).to(device)
                out_im, z_blank_im, _ = model(blank_im)
                loss = criterion(outputs_cam1, z_cam1, outputs_cam2, z_cam2, images_cam1, images_cam2, z_blank_im)
        else:
            loss = mse_1 + mse_2
            #loss = criterion(outputs_cam1, z_cam1, outputs_cam2, z_cam2, images_cam1, images_cam2)
            #loss = criterion(outputs_cam1, images_cam1) + criterion(outputs_cam2, images_cam2)
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
            images_cam1, images_cam2 = data
            images_cam1 = images_cam1.to(device)
            images_cam2 = images_cam2.to(device)
            # get model outputs
            outputs_cam1, z_cam1 = model(images_cam1)
            outputs_cam2, z_cam2 = model(images_cam2)
            loss = criterion(outputs_cam1, z_cam1, outputs_cam2, z_cam2, images_cam1, images_cam2)
            #loss = criterion(outputs_cam1, images_cam1) + criterion(outputs_cam2, images_cam2)
            val_loss += loss.item()  # * images_cam1.size(0)
        model.train()
        val_loss = val_loss / len(val_dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f}\tVal Loss: {:.6f}\tEpoch Time: {:.6f}'
              .format(epoch, train_loss,val_loss, time.time() - epoch_start_time), end="\r")
    else:
        print('Epoch: {} \tTraining Loss: {:.6f}\tEpoch Time: {:.6f}'.format(epoch, train_loss, time.time() - epoch_start_time))

torch.save(model.state_dict(), 'model_weights.pth')
print("Done!")





import PIL.Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StereoDataset
import time
import torchvision
from losses import MSE_and_Contrastive_loss, L1_and_Contrastive_loss, MSE_and_pairHamming_loss, L1_and_pairHamming_loss
from model_new import *
from model import *
from model_small import ImageCompressor_small


############## Train parameters ##############
#train_folder1 = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_02'
#train_folder2 = '/home/access/dev/data_sets/kitti/data_stereo_flow_multiview/train_small_set_8/image_03'
#train_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/diff_image_2'
#train_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/diff_image_2'

train_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_2'
train_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/training/image_3'
val_folder1 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_2'
val_folder2 = '/home/access/dev/data_sets/kitti/flow_2015/data_scene_flow/testing/image_3'
batch_size = 4
lr_start = 1e-4
epoch_patience = 150
n_epochs = 25000
val_every = 25000
save_every = 2000
useOnlyStereoPairsDistance = False
hammingLossOnBinaryZ = False
start_from_pretrained = '/home/access/dev/iclr_17_compression/checkpoints_new/not-binary-try/rec+dist (start from rec pretrained)/iter_637.pth.tar'
save_path = '/home/access/dev/iclr_17_compression/checkpoints_new/not-binary-try/rec+dist (start from rec pretrained)/decoder-only'

################ Data transforms ################
#tsfm = transforms.Compose([transforms.Resize((384, 1216), interpolation=3), transforms.ToTensor()])
tsfm = transforms.Compose([transforms.Resize((384, 1248), interpolation=3), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.Resize((192, 624), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.Resize((96, 320), interpolation=PIL.Image.BICUBIC), transforms.ToTensor()])
#tsfm = transforms.Compose([transforms.ToTensor()])


######### Set Seeds ###########
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


training_data = StereoDataset(stereo1_dir=train_folder1, stereo2_dir=train_folder2, randomFlip=False, RandomCrop=True, transform=tsfm)
val_data = StereoDataset(stereo1_dir=val_folder1, stereo2_dir=val_folder2, transform=tsfm)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load model:
#model = ImageCompressor_new(out_channel_N=256)
#model = ImageCompressor_new(out_channel_N=128)
#model = ImageCompressor_new()
model = ImageCompressor()
if start_from_pretrained != '':
    global_step_ignore = load_model(model, start_from_pretrained)
model = model.to(device)
encoder = model.Encoder
#model2 = ImageCompressor_new(out_channel_N=128)
model2 = ImageCompressor()
model_decoder = model2.Decoder.to(device)
model_decoder.train()

# todo: check with & without
#clipping_value = 5.0
#torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

criterion = nn.MSELoss()
#criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model_decoder.parameters(), lr=lr_start)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr_start, weight_decay=1e-8, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epoch_patience, verbose=True)

# Epochs
best_loss = 100
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    # Training
    epoch_start_time = time.time()
    for batch, data in enumerate(train_dataloader):
        images_cam1, images_cam2 = data
        images_cam1 = images_cam1.to(device)
        images_cam2 = images_cam2.to(device)

        optimizer.zero_grad()

        # Encoder (val only):
        z_cam1 = encoder(images_cam1)
        z_cam2 = encoder(images_cam2)
        # Add noise for binarization
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(z_cam1), -0.5, 0.5)
        # Decoder (training part of the network):
        outputs_cam1 = model_decoder(z_cam1 + quant_noise_feature)
        outputs_cam2 = model_decoder(z_cam2 + quant_noise_feature)
        outputs_cam1 = outputs_cam1.clamp(0., 1.)
        outputs_cam2 = outputs_cam2.clamp(0., 1.)

        loss = criterion(outputs_cam1, images_cam1) + criterion(outputs_cam2, images_cam2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() #* images_cam1.size(0)

    train_loss = train_loss / len(train_dataloader)
    # Note that step should be called after validate()
    scheduler.step(train_loss)
    if train_loss < best_loss:
        best_loss = train_loss
        save_model(model_decoder, epoch, save_path)#torch.save(model.state_dict(), '../model_best_weights.pth')
    elif epoch % save_every == 0:
        save_model(model_decoder, epoch, save_path)


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
              .format(epoch, train_loss,val_loss, time.time() - epoch_start_time))
    else:
        print('Epoch: {} \tTraining Loss: {:.6f}\tEpoch Time: {:.6f}'.format(epoch, train_loss, time.time() - epoch_start_time))

torch.save(model.state_dict(), 'model_weights.pth')
print("Done!")





from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from data_loader.datasets import Dataset
import torch
import pdb
from utils.image_utils import randFlipStereoImage
import glob


class Datasets(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob.glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    test_dataset = Datasets(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def get_train_loader(train_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True)
    return train_dataset, train_loader

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)

class StereoDataset(Dataset):
    """Stereo Image Pairs dataset."""
    def __init__(self, stereo1_dir, stereo2_dir, randomFlip=False, RandomCrop=False, crop_352_1216=False, transform=transforms.ToTensor()):
        """
        Args:
            stereo1_dir (string): Directory with all the images.
            stereo2_dir (string): Directory with all the images, from the second camera.
                note1: matching images from the two folders are assumed to have the same names.
                note2: assumes *png* images
            transform (optional): Optional transform to be applied on the images.
        """
        self.stereo1_dir = stereo1_dir
        self.stereo2_dir = stereo2_dir
        self.stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))
        self.transform = transform
        self.randomFlip = randomFlip
        self.RandomCrop = RandomCrop
        self.crop_352_1216 = crop_352_1216

    def __len__(self):
        return len(self.stereo1_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_stereo1 = Image.open(self.stereo1_path_list[idx])
        img_stereo2_name = os.path.join(self.stereo2_dir, os.path.basename(self.stereo1_path_list[idx]))
        try:
            img_stereo2 = Image.open(img_stereo2_name)
        except ValueError:
            raise ValueError("Error when reading stereo2-image. Image names in both folder should be the same.")

        if self.transform:
            img_stereo1 = self.transform(img_stereo1)
            img_stereo2 = self.transform(img_stereo2)

        if self.RandomCrop:
            #i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(320, 320))
            #i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(352, 1216))
            i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(370, 740))
            img_stereo1 = img_stereo1[:, i:i+h, j:j+w]
            img_stereo2 = img_stereo2[:, i:i+h, j:j+w]

            # temp patch: resize to fit Stereo paper dataset
            tsfm = transforms.Compose([transforms.Resize((128, 256))])
            img_stereo1 = tsfm(img_stereo1)
            img_stereo2 = tsfm(img_stereo2)

        if self.randomFlip:
            # convert to numpy, do augmentation, convert back to tensor
            im1_np = img_stereo1.permute(1, 2, 0).detach().cpu().numpy()
            im2_np = img_stereo2.permute(1, 2, 0).detach().cpu().numpy()
            img_stereo1, img_stereo2 = randFlipStereoImage(im1_np, im2_np)
            img_stereo1 = torch.tensor(img_stereo1.copy()).permute(2, 0, 1)
            img_stereo2 = torch.tensor(img_stereo2.copy()).permute(2, 0, 1)

        if self.crop_352_1216:
            # cut to the same size:
            img_stereo1 = img_stereo1[:, :352, :1216]
            img_stereo2 = img_stereo2[:, :352, :1216]


        return img_stereo1, img_stereo2


####
class StereoDataset_new(Dataset):
    """Stereo Image Pairs dataset."""
    def __init__(self, stereo_dir_2012, stereo_dir_2015, isTrainingData=True, randomFlip=False, RandomCrop=False, crop_352_1216=False, transform=transforms.ToTensor()):
        """
        Args:
            stereo_dir_2012 (string): Directory with stereo images from 2012 kitti dataset.
            stereo_dir_2015 (string): Directory with stereo images from 2015 kitti dataset.
                note1: inside each stereo_dir, expected sub folders: testing & training, and sub-sub folders image_2,image_3
                note2: matching images from the two folders(_2, _3) are assumed to have the same names.
                note3: assumes *png* images
            transform (optional): Optional transform to be applied on the images.
        """
        subFolder = 'training' if isTrainingData else 'testing'
        stereo2012_dir = os.path.join(stereo_dir_2012, subFolder, 'image_2')
        stereo2015_dir = os.path.join(stereo_dir_2015, subFolder, 'image_2')
        if isTrainingData:
            stereo2012_path_list = glob.glob(os.path.join(stereo2012_dir, '*png'))
            stereo2015_path_list = glob.glob(os.path.join(stereo2015_dir, '*png'))
        else:
            stereo2012_path_list = glob.glob(os.path.join(stereo2012_dir, '*10.png'))
            stereo2015_path_list = glob.glob(os.path.join(stereo2015_dir, '*10.png'))
        self.stereo_image_2_path_list = stereo2012_path_list + stereo2015_path_list
        self.transform = transform
        self.randomFlip = randomFlip
        self.RandomCrop = RandomCrop
        self.crop_352_1216 = crop_352_1216

    def __len__(self):
        return len(self.stereo_image_2_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_stereo1 = Image.open(self.stereo_image_2_path_list[idx])
        img_stereo2_name = self.stereo_image_2_path_list[idx].replace('image_2', 'image_3')
        try:
            img_stereo2 = Image.open(img_stereo2_name)
        except ValueError:
            raise ValueError("Error when reading stereo2-image. Image names in both folder should be the same.")

        if self.transform:
            img_stereo1 = self.transform(img_stereo1)
            img_stereo2 = self.transform(img_stereo2)

        if self.RandomCrop:
            i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(352, 1216))  # multiplication of 32
            img_stereo1 = img_stereo1[:, i:i+h, j:j+w]
            img_stereo2 = img_stereo2[:, i:i+h, j:j+w]

        if self.randomFlip:
            # convert to numpy, do augmentation, convert back to tensor
            im1_np = img_stereo1.permute(1, 2, 0).detach().cpu().numpy()
            im2_np = img_stereo2.permute(1, 2, 0).detach().cpu().numpy()
            img_stereo1, img_stereo2 = randFlipStereoImage(im1_np, im2_np)
            img_stereo1 = torch.tensor(img_stereo1.copy()).permute(2, 0, 1)
            img_stereo2 = torch.tensor(img_stereo2.copy()).permute(2, 0, 1)

        if self.crop_352_1216:
            # cut to the same size:
            img_stereo1 = img_stereo1[:, :352, :1216]
            img_stereo2 = img_stereo2[:, :352, :1216]


        return img_stereo1, img_stereo2

####
class StereoPlusDataset(Dataset):
    """Stereo Image Pairs dataset + added  image for contrastive learning."""
    def __init__(self, stereo1_dir, stereo2_dir, contrast_dir, RandomCrop=False, transform=transforms.ToTensor()):
        """
        Args:
            stereo1_dir (string): Directory with all the images.
            stereo2_dir (string): Directory with all the images, from the second camera.
                note1: matching images from the two folders are assumed to have the same names.
                note2: assumes *png* images
            transform (optional): Optional transform to be applied on the images.
        """
        self.stereo1_dir = stereo1_dir
        self.stereo2_dir = stereo2_dir
        self.stereo1_path_list = glob.glob(os.path.join(stereo1_dir, '*png'))
        self.contrast_path_list = glob.glob(os.path.join(contrast_dir, '*png'))
        self.transform = transform
        self.transforms_cont = transforms.Compose([transforms.RandomResizedCrop(368), transforms.ToTensor()])
        self.RandomCrop = RandomCrop

    def __len__(self):
        return len(self.stereo1_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_stereo1 = Image.open(self.stereo1_path_list[idx])
        img_stereo2_name = os.path.join(self.stereo2_dir, os.path.basename(self.stereo1_path_list[idx]))
        try:
            img_stereo2 = Image.open(img_stereo2_name)
        except ValueError:
            raise ValueError("Error when reading stereo2-image. Image names in both folder should be the same.")

        img_contrast = Image.open(self.contrast_path_list[idx])
        if self.transform:
            img_stereo1 = self.transform(img_stereo1)
            img_stereo2 = self.transform(img_stereo2)
            img_contrast = self.transforms_cont(img_contrast)

        if self.RandomCrop:
            i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(368, 368))
            img_stereo1 = img_stereo1[:, i:i+h, j:j+w]
            img_stereo2 = img_stereo2[:, i:i+h, j:j+w]


        return img_stereo1, img_stereo2, img_contrast

def build_dataset():
    train_set_dir = '/data1/liujiaheng/data/compression/Flick_patch/'
    dataset, dataloader = get_train_loader(train_set_dir, 256, 4)
    for batch_idx, (image, path) in enumerate(dataloader):
        pdb.set_trace()


if __name__ == '__main__':
    build_dataset()

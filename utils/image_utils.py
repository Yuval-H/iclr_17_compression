import torch
import numpy as np
from PIL import Image





def randFlipStereoImage(img1, img2):
    # Determine which kind of augmentation takes place according to probabilities
    random_chooser_lr = np.random.rand()
    random_chooser_ud = np.random.rand()
    if 0:#random_chooser_lr > 0.5:
        img1 = np.fliplr(img1)
        img2 = np.fliplr(img2)
    if random_chooser_ud > 0.5:
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    return img1, img2
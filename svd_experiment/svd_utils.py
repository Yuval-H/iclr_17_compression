import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from PIL import Image



def decompose_i_svd_components(z_x_down, i):
    # reshape to 2d array
    z_x_down = torch.squeeze(z_x_down)
    h,w,c = z_x_down.size()
    z_x_down = torch.reshape(z_x_down, (h*w, c))
    # perform SVD decomposition
    z_x_down = z_x_down.cpu().detach().numpy()
    U, sigma, V = np.linalg.svd(z_x_down)
    return np.matrix(U[:, :i]), np.diag(sigma[:i]), np.matrix(V[:i, :]), h,w,c


def compose_Z_x_down_from_svd_components(u, sigma, v, h,w,c):
    recon = u * sigma * v
    recon = torch.from_numpy(recon).float().to('cuda')
    recon = (torch.round(recon/16)*16)
    recon = torch.reshape(recon, (h,w,c))
    return torch.unsqueeze(recon,0)


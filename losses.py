import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_loss_pairsOnly(nn.Module):
    def __init__(self, margin=0.01):
        super(Contrastive_loss_pairsOnly, self).__init__()
        self.margin = margin
        self.mseLoss = nn.MSELoss()

    def forward(self, z1, z2):

        size = z2.size()
        z1 = z1.view(z1.size()[0], -1)
        z2 = z2.view(z2.size()[0], -1)
        code_len = z2.size()[1]

        similarity_loss = 0
        for iCode in range(z1.size()[0]):
            #hamm_dist = torch.mean(torch.square(z1[iCode,:]-z2[iCode,:])) # replace mean to sum* (1/code_len)
            hamm_dist = self.mseLoss(z1[iCode,:], z2[iCode,:])
            if hamm_dist > self.margin:
                similarity_loss = similarity_loss + (hamm_dist - self.margin)
        similarity_loss = similarity_loss / z1.size()[0]
        return similarity_loss


class Contrastive_loss(nn.Module):
    def __init__(self, margin_neg=0.4, margin_pos=0.01):
        super(Contrastive_loss, self).__init__()
        self.margin_neg = margin_neg
        self.margin_pos = margin_pos
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        #self.NPairsLoss = losses.NPairsLoss()
        #distances.LpDistance()
        self.mseLoss = nn.MSELoss()
        self.l1Loss = nn.L1Loss()

    def forward(self, z1, z2):

        # similarity_loss = nn.MSELoss(z1, z2)
        size = z2.size()
        z1 = z1.view(z1.size()[0], -1)
        z2 = z2.view(z2.size()[0], -1)
        code_len = z2.size()[1]

        ########################################################
        similarity_lossP = 0
        similarity_lossN = 0
        for m in range(z1.size()[0]):
            for n in range(z2.size()[0]):
                hamm_dist = self.mseLoss(z1[m,:], z2[n,:])  # torch.sum(torch.square(z1[m,:]-z2[n,:])) * (1/code_len)
                #hamm_dist = self.l1Loss(z1[m,:], z2[n,:])
                if m == n:
                    if hamm_dist > self.margin_pos:
                        similarity_lossP = similarity_lossP + (hamm_dist - self.margin_pos)
                else:
                    if hamm_dist < self.margin_neg:
                        similarity_lossN = similarity_lossN + (self.margin_neg -hamm_dist)
        similarity_lossP = similarity_lossP / z1.size()[0]
        if z2.size()[0] > 1: #avoid zero division
            similarity_lossN = similarity_lossN / (z1.size()[0]*(z2.size()[0]-1))
        #similarity_loss = similarity_lossP + similarity_lossN
        similarity_loss = 1.5*similarity_lossP + 0.5*similarity_lossN

        return similarity_loss



class MSE_and_pairHamming_loss(nn.Module):
    def __init__(self, eps=0.1, margin=0.01, p_one=0.45):
        super(MSE_and_pairHamming_loss, self).__init__()
        self.eps = eps
        self.margin = margin
        self.p_one = p_one
        self.mse = nn.MSELoss()

    def forward(self, outputs1, z1, outputs2, z2, input1, input2):
        reconstruction_loss = torch.mean((outputs1-input1).pow(2)) + torch.mean((outputs2-input2).pow(2))  #self.mse(outputs1, input1) + self.mse(outputs2, input2)
        # Hamming distance and codes-p
        pairHaaming_loss = torch.mean((z1-z2).pow(2))  #self.mse(z1,z2)
        if pairHaaming_loss > self.margin:
            pairHaaming_loss = pairHaaming_loss - self.margin
        # # Make sure code is not all zeros
        # p_one = 0.5*torch.mean(z1.pow(2)) + 0.5*torch.mean(z2.pow(2))
        # p_one_loss = torch.sqrt((p_one - self.p_one).pow(2))
        # code_loss = 0.7*pairHaaming_loss + 0.3*p_one_loss
        code_loss = pairHaaming_loss
        return reconstruction_loss + self.eps*code_loss

class MSE_and_Contrastive_loss(nn.Module):
    def __init__(self, eps=1, margin=0.3, useOnlyPair=False):
        super(MSE_and_Contrastive_loss, self).__init__()
        self.eps = eps
        self.margin = margin
        self.mse = nn.MSELoss()
        self.contrastive_loss = Contrastive_loss()
        self.contrastive_pair_loss = Contrastive_loss_pairsOnly()
        self.useOnlyPair = useOnlyPair

    def forward(self, outputs1, z1, outputs2, z2, input1, input2):
        reconstruction_loss = self.mse(outputs1, input1) + self.mse(outputs2, input2)
        if self.useOnlyPair:
            similarity_loss = self.contrastive_pair_loss(z1, z2)
        else:
            similarity_loss = self.contrastive_loss(z1, z2)
        #similarity_loss = 0
        #if self.eps*similarity_loss > reconstruction_loss:
        #    print('recon_loss=', reconstruction_loss,", contrastive_loss = ",similarity_loss)
        #ratio = reconstruction_loss/similarity_loss
        return reconstruction_loss + self.eps*similarity_loss

class L1_and_Contrastive_loss(nn.Module):
    def __init__(self, eps=1, margin=0.3):
        super(L1_and_Contrastive_loss, self).__init__()
        self.eps = eps
        self.margin = margin
        self.L1loss = nn.L1Loss()
        self.contrastive_loss = Contrastive_loss()
        self.contrastive_pair_loss = Contrastive_loss_pairsOnly()

    def forward(self, outputs1, z1, outputs2, z2, input1, input2):
        reconstruction_loss = self.L1loss(outputs1, input1) + self.L1loss(outputs2, input2)
        similarity_loss = self.contrastive_loss(z1, z2)
        #similarity_loss = self.contrastive_pair_loss(z1, z2)

        return reconstruction_loss + self.eps*similarity_loss





#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class EdgeAndCharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(EdgeAndCharbonnierLoss, self).__init__()
        self.eps = eps
        self.loss_edge = EdgeLoss()
        self.loss_char = CharbonnierLoss()

    def forward(self, x, y):
        loss = self.loss_char.forward(x,y) + 0.05 * self.loss_edge.forward(x,y)
        # a = 1 - ms_ssim(x, y, data_range=255, size_average=True)
        #loss = 1 - MultiScaleSSIM(x, y)

        return loss


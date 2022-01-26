import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)



class BottleneckAttention(nn.Module):
    def __init__(
            self,
            dim,
            fmap_size,
            heads=4,
            dim_head=None,

    ):
        super().__init__()
        self.heads = heads
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.scale = self.dim_head ** -0.5
        self.fmap_size = fmap_size
        self.to_q = nn.Conv2d(dim, heads * self.dim_head, 1, bias=False)
        self.to_k = nn.Conv2d(dim, heads * self.dim_head, 1, bias=False)

        self.height = self.fmap_size[0]
        self.width = self.fmap_size[1]


    def forward(self, z1_hat, z2):
        assert z1_hat.dim() == 4, f'Expected 4D tensor, got {z1_hat.dim()}D tensor'

        q = z1_hat#self.to_q(z1_hat)
        k = z2#self.to_k(z2)
        v = z2
        # [batch (heads*3*dim_head) height width]
        qkv = torch.cat((q, k, v), 1)
        # decompose heads and merge spatial dims as tokens
        q, k, v = tuple((rearrange(qkv, 'b (k d) h w  -> k b (h w) d', k=3)))  #tuple(rearrange(qkv, 'b (d k H ) h w  -> k b H (h w) d', k=3, H=self.heads))

        # i, j refer to tokens
        dot_prod = einsum('b i d, b j d -> b i j', q, k) * self.scale #einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attention = torch.softmax(dot_prod, dim=-1)
        out = einsum('b i j, b j d -> b i d', attention, v) #einsum('b h i j, b h j d -> b h i d', attention, v)
        # Merge heads and decompose tokens to spatial dims
        out = rearrange(out, 'b (x y) d -> b d x y', x=self.height, y=self.width) #rearrange(out, 'b h (x y) d -> b (h d) x y', x=self.height, y=self.width)
        return out

class ResidualBlock_3_3(nn.Module):
    """Simple residual block with 3x3, 1*1 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1)
        #if in_ch != out_ch:
        #    self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        #else:
        #    self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        #if self.skip is not None:
        #    identity = self.skip(x)

        #out = out + identity
        return out

class BottleneckAttention_modified(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=None
    ):
        super().__init__()
        self.dim_head = dim if dim_head is None else dim_head
        self.dim = dim
        self.scale = self.dim_head ** -0.5
        self.to_q = nn.Sequential(nn.Conv2d(dim, self.dim_head, 1), nn.LeakyReLU(),  nn.Conv2d(self.dim_head, self.dim_head, 1),nn.LeakyReLU())
        self.to_k = nn.Sequential(nn.Conv2d(dim, self.dim_head, 1), nn.LeakyReLU(),  nn.Conv2d(self.dim_head, self.dim_head, 1),nn.LeakyReLU())
        #self.to_q = nn.Conv2d(dim, self.dim_head, kernel_size=(16, 16), stride=(16, 16))
        #self.to_k = nn.Conv2d(dim, self.dim_head, kernel_size=(16, 16), stride=(16, 16))

        #self.to_base_tokens = nn.Sequential(ResidualBlock_3_3(dim, self.dim_head),
        #                                    ResidualBlock_3_3(self.dim_head, self.dim_head),
        #                                    ResidualBlock_3_3(self.dim_head, self.dim_head))

        #self.to_q = ResidualBlock_3_3(self.dim_head, self.dim_head, stride=9)#nn.Conv2d(self.dim_head, self.dim_head, kernel_size=3, stride=9)

        #self.to_k = ResidualBlock_3_3(self.dim_head, self.dim_head, stride=4)#nn.Conv2d(self.dim_head, self.dim_head, kernel_size=3, stride=4)





    def forward(self, Q, K, V):
        assert Q.dim() == 4, f'Expected 4D tensor, got {Q.dim()}D tensor'

        # non overlapping 9*9
        q = self.to_q(Q)#self.to_q(self.to_base_tokens(Q))
        # overlapping 9*9 with stride=4
        k = self.to_k(K)#self.to_k(self.to_base_tokens(K))
        v = V
        # [batch (3*dim_head) height width]
        #qkv = torch.cat((q, k, v), 1)
        # decompose heads and merge spatial dims as tokens
        # each of size [(h*w) dim_head]
        #q, k, v = tuple((rearrange(qkv, 'b (k d) h w  -> k b (h w) d', k=3)))
        #q = torch.squeeze(q)
        #k = torch.squeeze(k)
        #v = torch.squeeze(v)

        # Q
        # patch size
        kc, kh, kw = self.dim_head, 1, 1 #self.dim_head, 40, 76 #128, 10, 4 #128, 10, 10
        # stride Q (non overlapping)
        dc, dh, dw = self.dim_head, 1, 1 #self.dim_head, 40, 76 #128, 10, 4 #128, 10, 10

        #x=q
        #q = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
        #              x.size(1) % kh // 2, x.size(1) % kh // 2,
        #              x.size(0) % kc // 2, x.size(0) % kc // 2))

        patches = q.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape_q = list(patches.size())#patches.size()
        patches = patches.contiguous().view(-1, kc, kh, kw)
        q = rearrange(patches, 'p c h w  -> p (c h w)')
        # K and V, choose patch size and stride (overlapping is an option)
        #kc, kh, kw = 1, 10, 38#self.dim_head, 40, 76
        dc, dh, dw = 1,1,1#self.dim_head, 4, 8
        # K
        #x = k
        #k = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
        #              x.size(1) % kh // 2, x.size(1) % kh // 2,
        #              x.size(0) % kc // 2, x.size(0) % kc // 2))
        patches = k.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(-1, kc, kh, kw)
        k = rearrange(patches, 'p c h w  -> p (c h w)')
        # V
        kc, kh, kw = 3, 9, 9#self.dim, 40, 76
        dc, dh, dw = 3,4,4#self.dim, 4, 8
        #x = v
        #v = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
        #              x.size(1) % kh // 2, x.size(1) % kh // 2,
        #              x.size(0) % kc // 2, x.size(0) % kc // 2))
        patches = v.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = list(patches.size())
        #fix for the current number of Q
        unfold_shape[2:4] = unfold_shape_q[2:4]
        patches = patches.contiguous().view(-1, kc, kh, kw)
        v = rearrange(patches, 'p c h w  -> p (c h w)')
        # clear patches memory
        patches = 0


        k_transpose = torch.transpose(k, 0, 1)
        # dot product. attention are of size [(h*w) (h*w)] -> ~ each feature of z1_hat with respect to z2
        dot_prod = torch.matmul(q, k_transpose) * (k_transpose.size()[0] ** -1) #self.scale
        ####attention = F.normalize(dot_prod, p=2, dim=-1)
        ####attention = attention.pow(2)
        attention = torch.softmax(dot_prod, dim=-1)
        # out, [(h*w) dim]
        out = torch.matmul(attention, v)

        pa = rearrange(out, 'p (c h w)  -> p c h w', c=kc, h=kh, w=kw)

        # combine patches
        patches_orig = pa.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        patches_orig = patches_orig.view(1, output_c, output_h, output_w)

        return patches_orig



class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

import torch
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F



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



class BottleneckAttention_modified(nn.Module):
    def __init__(
            self,
            dim,
            fmap_size,
            dim_head=None
    ):
        super().__init__()
        self.dim_head = dim if dim_head is None else dim_head
        self.scale = self.dim_head ** -0.5
        self.fmap_size = fmap_size
        self.to_q = nn.Conv2d(dim, self.dim_head, 1, bias=False)
        self.to_k = nn.Conv2d(dim, self.dim_head, 1, bias=False)

        self.height = self.fmap_size[0]
        self.width = self.fmap_size[1]


    def forward(self, z1_hat, z2):
        assert z1_hat.dim() == 4, f'Expected 4D tensor, got {z1_hat.dim()}D tensor'

        q = z1_hat#self.to_q(z1_hat)
        k = z2#self.to_k(z2)
        v = z2
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
        kc, kh, kw = 128, 10, 4
        # stride Q (non overlapping)
        dc, dh, dw = 128, 10, 4

        #x=q
        #q = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
        #              x.size(1) % kh // 2, x.size(1) % kh // 2,
        #              x.size(0) % kc // 2, x.size(0) % kc // 2))

        patches = q.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, kc, kh, kw)
        q = rearrange(patches, 'p c h w  -> p (c h w)')
        # K and V, choose stride (overlapping is an option)
        dc, dh, dw = 128, 1, 1
        # K
        #x = k
        #k = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
        #              x.size(1) % kh // 2, x.size(1) % kh // 2,
        #              x.size(0) % kc // 2, x.size(0) % kc // 2))
        patches = k.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(-1, kc, kh, kw)
        k = rearrange(patches, 'p c h w  -> p (c h w)')
        # V
        #x = v
        #v = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
        #              x.size(1) % kh // 2, x.size(1) % kh // 2,
        #              x.size(0) % kc // 2, x.size(0) % kc // 2))
        patches = v.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(-1, kc, kh, kw)
        v = rearrange(patches, 'p c h w  -> p (c h w)')


        k_transpose = torch.transpose(k, 0, 1)
        # dot product. attention are of size [(h*w) (h*w)] -> ~ each feature of z1_hat with respect to z2
        dot_prod = torch.matmul(q, k_transpose) * (k_transpose.size()[0] ** -1)#self.scale
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

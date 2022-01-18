import torch
from einops import rearrange
from torch import nn, einsum



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

        q = self.to_q(z1_hat)
        k = self.to_k(z2)
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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from .refine import Refine
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
from rotary_embedding_torch import RotaryEmbedding

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaled dot-product attention
        self.rope=RotaryEmbedding(self.head_dim).cuda().rotate_queries_or_keys
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        # B, HW, T, C = x.shape  # (B * H * W, T, C)
        
        # Apply linear projections for query, key, and value
        q = self.q_proj(x)  # (B * H * W, T, C)
        k = self.k_proj(x)  # (B * H * W, T, C)
        v = self.v_proj(x)  # (B * H * W, T, C)
        
        # Reshape into multi-head format (B * H * W, T, num_heads, head_dim)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads, d=self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
        assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Last dimension (head_dim) must match"
        assert q.dtype == torch.float32, "dtype of q, k, v must be torch.float32"

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = rearrange(attn_output,'b h t d->b t (h d)')
        # Project the output back to original dimension (dim)

        out = self.out_proj(attn_output)
        
        return out


class CrossTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossTemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaled dot-product attention
        self.rope=RotaryEmbedding(self.head_dim).cuda().rotate_queries_or_keys
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x,c):
        # B, HW, T, C = x.shape  # (B * H * W, T, C)
        
        # Apply linear projections for query, key, and value
        q = self.q_proj(x)  # (B * H * W, T, C)
        k = self.k_proj(c)  # (B * H * W, T, C)
        v = self.v_proj(c)  # (B * H * W, T, C)
        
        # Reshape into multi-head format (B * H * W, T, num_heads, head_dim)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads, d=self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        # Apply RoPE (Rotary Positional Encoding) to queries and keys
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=None,  # Optional: provide an attention mask if necessary
            dropout_p=0.0,  # Optional: dropout rate during attention computation
            is_causal=False  # Set to True for autoregressive (causal) attention
        )
        attn_output = rearrange(attn_output,'b h t d->b t (h d)')
        # Project the output back to original dimension (dim)
        out = self.out_proj(attn_output)
        
        return out


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.sa = SelfAttention(dim,num_heads,bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        # ms = ms+self.sa(self.norm(ms))
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms

class SpatialTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SpatialTemporalAttention, self).__init__()
        self.spatial_attention = SingleTransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.temporal_attention = TemporalTransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
    def forward(self,x,t):
        #x.shape:((b t) c h w)
        x = self.spatial_attention(x)
        x = self.temporal_attention(x,t)
        return x

class SingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SingleTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.attn = SelfAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, x):
        x = x+self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TemporalTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.attn = TemporalAttention(dim,num_heads,bias)
    def forward(self, x,T):
        norm_x = self.norm1(x)
        BT,C,H,W=norm_x.shape
        norm_x=rearrange(norm_x,'(B T) C H W -> (B H W) T C',T=T)
        x_attn = self.attn(norm_x)
        x_attn = rearrange(x_attn,'(B H W) T C->(B T) C H W',H=H,W=W)
        x = x+x_attn
        x = x + self.ffn(self.norm2(x))
        return x

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.attn = CrossTemporalAttention(dim,num_heads,bias)
    def forward(self, x,c,T):
        norm_x = self.norm1(x)
        c = self.norm3(c)
        BT,C,H,W=norm_x.shape
        norm_x=rearrange(norm_x,'(B T) C H W -> (B H W) T C',T=T)
        c=rearrange(c,'(B T) C H W -> (B H W) T C',T=T)
        x_attn = self.attn(norm_x,c)
        x_attn = rearrange(x_attn,'(B H W) T C->(B T) C H W',H=H,W=W)
        x = x+x_attn
        x = x + self.ffn(self.norm2(x))
        return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)

class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi
        

def patchfy(img, patch_size):
    """
    将输入图像分割成 patches
    """
    b, c, h, w = img.shape
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(b, c, -1, patch_size, patch_size)  # 将补丁拉平
    patches = patches.permute(0, 2, 1, 3, 4)  # 变换维度 (b, t, c, patch_size, patch_size)
    return patches

def unpatchfy(patches, img_shape, patch_size):
    """
    将 patches 还原回原始图像
    """
    b, t, c, ph, pw = patches.shape
    h, w = img_shape[2], img_shape[3]
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    # 重新排列 patches
    patches = patches.permute(0, 2, 1, 3, 4)  # 变回 (b, c, t, patch_size, patch_size)
    patches = patches.contiguous().view(b, c, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()  # 调整维度顺序
    img_reconstructed = patches.view(b, c, h, w)  # 还原形状
    return img_reconstructed
    
class Net(nn.Module):
    def __init__(self,num_channels=None,base_filter=None, scale = 4,args=None):
        super(Net, self).__init__()
        base_filter=32
        self.base_filter = base_filter
        self.scale = scale
        self.stride=1
        self.patch_size=1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4,base_filter,3,1,1),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter),HinResBlock(base_filter,base_filter))
        self.embed_dim = base_filter*self.stride*self.patch_size
        self.deep_fusion1 = TransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        self.deep_fusion2 = TransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        self.deep_fusion3 = TransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        self.deep_fusion4 = TransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        self.deep_fusion5 = TransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        self.temporal_fusion1 = CrossTransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        self.temporal_fusion2 = CrossTransformerBlock(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias')
        
        self.pan_feature_extraction = nn.ModuleList([SpatialTemporalAttention(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias') for i in range(4)])
        self.ms_feature_extraction = nn.ModuleList([SpatialTemporalAttention(base_filter,4,bias = False,ffn_expansion_factor=2.66,LayerNorm_type = 'WithBias') for i in range(4)])

        self.output = Refine(base_filter,4)
        self.to_patch=True
    def forward(self,ms,_,pan):
        _,_,h_in,w_in = pan.shape 

        patch_sizes = [h_in, h_in//2, h_in//4, h_in//8]
        probabilities = [0.2, 0.4, 0.2, 0.2]
        if self.training:  # Random patch size during training
            # Choose patch size based on the probabilities
            patch_size = torch.multinomial(torch.tensor(probabilities), 1).item()
            patch_size = patch_sizes[patch_size]
        else:  # Fixed patch size during testing (64)
            patch_size = 200

        ms_bic = F.interpolate(ms,scale_factor=self.scale)
        if self.to_patch:

            ms_bic_patch = patchfy(ms_bic,patch_size)
            pan_bic_patch = patchfy(pan,patch_size)
            b,t,c,h,w = ms_bic_patch.shape
            ms_bic_patch = rearrange(ms_bic_patch,'b t c h w->(b t) c h w')
            pan_bic_patch = rearrange(pan_bic_patch,'b t c h w->(b t) c h w')
                
            ms_f = self.ms_encoder(ms_bic_patch)
            pan_f = self.pan_encoder(pan_bic_patch)
            b,c,h,w = ms_f.shape
            for layer in self.pan_feature_extraction:
                pan_f = layer(pan_f,t)
            for layer in self.ms_feature_extraction:
                ms_f = layer(ms_f,t)

            ms_f = self.deep_fusion1(ms_f,pan_f)
            ms_f = self.temporal_fusion1(ms_f,pan_f,t)
            ms_f = self.deep_fusion2(ms_f,pan_f)
            ms_f = self.temporal_fusion2(ms_f,pan_f,t)
            ms_f = self.deep_fusion3(ms_f,pan_f)
            ms_f = self.deep_fusion4(ms_f,pan_f)
            ms_f = self.deep_fusion5(ms_f,pan_f)

            ms_f = self.output(ms_f)
            ms_f =  rearrange(ms_f,'(b t) c h w->b t c h w',t=t)

            hrms = unpatchfy(ms_f,(b,4,h_in,w_in),patch_size)+ms_bic
        else:
            ms_f = self.ms_encoder(ms_bic)
            pan_f = self.pan_encoder(pan)
            ms_f = self.ms_feature_extraction(ms_f)
            pan_f = self.pan_feature_extraction(pan_f)
            ms_f = self.deep_fusion1(ms_f,pan_f)
            ms_f = self.deep_fusion2(ms_f,pan_f)
            ms_f = self.deep_fusion3(ms_f,pan_f)

            # ms_f = ms_f+pan_f
            ms_f = self.output(ms_f)
            hrms=ms_f+ms_bic
        return hrms


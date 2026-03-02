import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from mamba_ssm.modules.mamba_simple import Mamba
from .refine import Refine

global input_size
input_size = 128 * 1

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, stride=4, in_chans=36, embed_dim=32 * 32 * 32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim, 'BiasFree')

    def forward(self, x):
        # （b,c,h,w)->(b,c*s*p,h//s,w//s)
        # (b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x


class RandomMambaExtraction(nn.Module):
    def __init__(self, dim, i):
        super(RandomMambaExtraction, self).__init__()
        self.f = RandomMamba(dim)

    def forward(self, ipt):
        return self.f(ipt)


import random


class RandomMambaOperator(nn.Module):
    def __init__(self, dim):
        super(RandomMambaOperator, self).__init__()
        self.encoder = Mamba(dim, bimamba_type=None)
        self.repeat = 4
        self.pos = nn.Conv2d(32, 32, 5, 1, 2, groups=32)
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)

    def forward(self, x):
        if not self.training:
            x = x.repeat(self.repeat, 1, 1)
        B, N, C = x.shape

        x_pos = self.pos(x.transpose(1, 2).view(B, 32, input_size, input_size)).flatten(2).transpose(1, 2)
        x = x + x_pos

        N_index = list(range(0, N))
        N_shuffle = list(range(0, N))
        N_shuffle_list = []
        shuffle_x = x.clone()

        for b in range(B):
            random.shuffle(N_shuffle)
            N_shuffle_list.append(N_shuffle)
            N_shuffle = list(range(0, N))

        for b, N_shuffle in enumerate(N_shuffle_list):
            shuffle_x[b, :, :] = x[b, N_shuffle, :]

        shuffle_x_out = self.encoder(shuffle_x)

        RN_shuffle_list = []
        for b in range(B):
            RN_shuffle_list.append([N_shuffle_list[b][i] for i in N_index])
        for b, rn_shuffle in enumerate(RN_shuffle_list):
            shuffle_x_out[b, rn_shuffle, :] = shuffle_x_out[b, N_index, :]
        out = shuffle_x_out
        if not self.training:
            out = torch.stack(out.chunk(self.repeat, dim=0), dim=0)
            out = torch.mean(out, dim=0)
        return out


class RandomMamba(nn.Module):
    def __init__(self, dim):
        super(RandomMamba, self).__init__()
        self.encoder = Mamba(dim, bimamba_type=None)
        self.norm = LayerNorm(dim, 'with_bias')
        self.repeat = 4
        self.pos = nn.Conv2d(32, 32, 5, 1, 2, groups=32)

    def forward(self, ipt):
        x, residual = ipt
        residual = x + residual
        x = self.norm(residual)

        if not self.training:
            x = x.repeat(self.repeat, 1, 1)

        B, N, C = x.shape

        x_pos = self.pos(x.transpose(1, 2).view(B, 32, input_size, input_size)).flatten(2).transpose(1, 2)
        x = x + x_pos

        indices = torch.randperm(N)

        # Shuffle x along the second dimension (N)
        shuffle_x = x[:, indices, :]

        # Encode shuffled x
        shuffle_x_out = self.encoder(shuffle_x)

        # Restore the original order of elements
        restored_x_out = shuffle_x_out[:, indices, :]

        if not self.training:
            restored_x_out = torch.mean(torch.stack(restored_x_out.chunk(self.repeat, dim=0), dim=0), dim=0)

        return restored_x_out, residual


# Random Channel Interactive Mamba
class RCIM(nn.Module):
    def __init__(self, dim):
        super(RCIM, self).__init__()
        self.msencoder = RandomMambaOperator(dim)
        self.panencoder = RandomMambaOperator(dim)
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')

    def forward(self, ms, pan
                , ms_residual, pan_residual):
        # ms (B,N,C)
        # pan (B,N,C)
        ms_residual = ms + ms_residual
        pan_residual = pan + pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B, N, C = ms.shape
        ms_first_half = ms[:, :, :C // 2]
        pan_first_half = pan[:, :, :C // 2]
        ms_swap = torch.cat([pan_first_half, ms[:, :, C // 2:]], dim=2)
        pan_swap = torch.cat([ms_first_half, pan[:, :, C // 2:]], dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap, pan_swap, ms_residual, pan_residual


import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomModalOperator(nn.Module):
    def __init__(self, dim):
        super(RandomModalOperator, self).__init__()
        self.encoder = Mamba(dim, bimamba_type="v3")
        self.repeat = 4
        self.pos_ms = nn.Conv2d(32, 32, 5, 1, 2, groups=32)
        self.pos_pan = nn.Conv2d(32, 32, 5, 1, 2, groups=32)

    def forward(self, ms, pan):
        if not self.training:
            ms = ms.repeat(self.repeat, 1, 1)
            pan = pan.repeat(self.repeat, 1, 1)
        B, N, C = ms.shape
        ms_pos = self.pos_ms(ms.transpose(1, 2).view(B, 32, input_size, input_size)).flatten(2).transpose(1, 2)
        pan_pos = self.pos_pan(ms.transpose(1, 2).view(B, 32, input_size, input_size)).flatten(2).transpose(1, 2)

        ms = ms + ms_pos
        pan = pan + pan_pos
        N_index = list(range(0, N))
        N_shuffle = list(range(0, N))
        N_shuffle_list = []
        shuffle_ms = ms.clone()
        shuffle_pan = pan.clone()
        for b in range(B):
            random.shuffle(N_shuffle)
            N_shuffle_list.append(N_shuffle)
            N_shuffle = list(range(0, N))
        for b, N_shuffle in enumerate(N_shuffle_list):
            shuffle_ms[b, :, :] = ms[b, N_shuffle, :]
            shuffle_pan[b, :, :] = pan[b, N_shuffle, :]
        shuffle_x_out = self.encoder(shuffle_ms, extra_emb=shuffle_pan)
        RN_shuffle_list = []
        for b in range(B):
            RN_shuffle_list.append([N_shuffle_list[b][i] for i in N_index])
        for b, rn_shuffle in enumerate(RN_shuffle_list):
            shuffle_x_out[b, rn_shuffle, :] = shuffle_x_out[b, N_index, :]
        out = shuffle_x_out
        if not self.training:
            out = torch.stack(out.chunk(self.repeat, dim=0), dim=0)
            out = torch.mean(out, dim=0)
        return out


class RandomModalMamba(nn.Module):
    def __init__(self, dim):
        super(RandomModalMamba, self).__init__()
        self.cross_mamba = RandomModalOperator(dim)
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, ms, ms_resi, pan):
        ms_resi = ms + ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        global_f = self.cross_mamba(self.norm1(ms), self.norm2(pan))
        B, HW, C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, input_size, input_size)
        ms = (self.dwconv(ms) + ms).flatten(2).transpose(1, 2)
        return ms, ms_resi


# Random Modal Interactive Mamba
class RMIM(nn.Module):
    def __init__(self, dim, i):
        super(RMIM, self).__init__()
        self.encoder = RandomModalMamba(dim)

    def forward(self, ms, ms_resi, pan):
        return self.encoder(ms, ms_resi, pan)


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
        return x + resi


class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        base_filter = 32
        self.base_filter = base_filter
        self.stride = 1
        self.patch_size = 1
        self.pan_encoder = nn.Sequential(nn.Conv2d(1, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                        HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.embed_dim = base_filter * self.stride * self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.shallow_fusion2 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.ms_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                      stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size,
                                       stride=self.stride)

        self.pan_feature_extraction = nn.Sequential(*[RandomMambaExtraction(self.embed_dim, i) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[RandomMambaExtraction(self.embed_dim, i) for i in range(8)])

        self.channelinteract_mamba1 = RCIM(self.embed_dim)
        self.channelinteract_mamba2 = RCIM(self.embed_dim)

        self.deep_fusion1 = RMIM(self.embed_dim, 0)
        self.deep_fusion2 = RMIM(self.embed_dim, 1)
        self.deep_fusion3 = RMIM(self.embed_dim, 2)
        self.deep_fusion4 = RMIM(self.embed_dim, 3)
        self.deep_fusion5 = RMIM(self.embed_dim, 4)

        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter, 4)

    def forward(self, ms, _, pan):
        ms_bic = F.interpolate(ms, scale_factor=4)
        ms_f = self.ms_encoder(ms_bic)
        # ms_f = ms_bic
        # pan_f = pan
        b, c, h, w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        residual_pan_f = 0
        ms_f, residual_ms_f = self.ms_feature_extraction([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction([pan_f, residual_pan_f])
        ms_f, pan_f, residual_ms_f, residual_pan_f = self.channelinteract_mamba1(ms_f, pan_f, residual_ms_f,
                                                                                 residual_pan_f)
        ms_f, pan_f, residual_ms_f, residual_pan_f = self.channelinteract_mamba2(ms_f, pan_f, residual_ms_f,
                                                                                 residual_pan_f)
        ms_f = self.patchunembe(ms_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        ms_f = self.shallow_fusion1(torch.concat([ms_f, pan_f], dim=1)) + ms_f
        pan_f = self.shallow_fusion2(torch.concat([pan_f, ms_f], dim=1)) + pan_f
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        ms_f, residual_ms_f = self.deep_fusion1(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion2(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion3(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion4(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion5(ms_f, residual_ms_f, pan_f)
        ms_f = self.patchunembe(ms_f, (h, w))
        hrms = self.output(ms_f) + ms_bic
        return hrms

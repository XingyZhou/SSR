"""HyMCNetL3_V7: HyMCNetL3 with the spatial Mamba scan replaced by a spectral-axis scan.

Single change relative to HyMCNetL3 (Mamba_Model.py): inside every MambaB the
Mamba_Att (raster scan over H*W pixels) becomes SpecMamba_Att. The backbone, the
Haar down/up-sampling, ConvB, FFN, losses and training recipe are untouched, so
HyMCNetL3 vs V7 isolates the scan axis.

SpecMamba_Att: the feature map is cut into non-overlapping win x win windows. Each
window gives one sequence over the C channels, and every channel is a token whose
feature vector is its win*win pixels (the same token layout SSR's WSSA uses for
spectral attention, but with a selective-scan Mamba instead of attention). Mamba
then scans along the channel / wavelength order, where neighbouring tokens are
physically adjacent bands. Spatial structure is left to the 5x5 pos conv, ConvB
(11x11 depthwise) and the U-Net's multi-scale path; windows are not mixed in the
scan (the 5x5 / 11x11 convs and the Haar levels carry information across them).
"""
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba

from Mamba_Model import (
    PreNorm,
    FFN,
    ConvB,
    MamConv_UNet_L3,
    _HyMCNetBase,
)


class SpecMamba_Att(nn.Module):
    def __init__(self, dim=28, win=8, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.win = win
        # d_model is the window area; the sequence length is the channel count `dim`
        self.attn = Mamba(d_model=win * win, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.win
        assert h % p == 0 and w % p == 0, "feature map must be divisible by the window size"
        t = rearrange(x, "b c (h p) (w q) -> (b h w) c (p q)", p=p, q=p)
        t = self.attn(t.contiguous())
        return rearrange(t, "(b h w) c (p q) -> b c (h p) (w q)", b=b, h=h // p, w=w // p, p=p, q=p)


class MambaB_V7(nn.Module):
    def __init__(self, dim, win=8):
        super().__init__()
        self.Mamba = PreNorm(dim, SpecMamba_Att(dim=dim, win=win))
        self.FFN = PreNorm(dim, FFN(dim=dim))

    def forward(self, x):
        x = self.Mamba(x) + x
        x = self.FFN(x) + x
        return x


class MamConvB_V7(nn.Module):
    """Same layout as MamConvB: 5x5 pos conv, MambaB, ConvB."""

    def __init__(self, dim, win=8):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.MambaB = MambaB_V7(dim, win)
        self.ConvB = ConvB(dim)

    def forward(self, x):
        x = self.pos(x) + x
        x = self.MambaB(x)
        x = self.ConvB(x)
        return x


class MamConv_UNet_L3_V7(MamConv_UNet_L3):
    """HyMCNetL3's U-Net (Haar down/up) with spectral-scan blocks."""

    def __init__(self, out_dim=28):
        super().__init__(out_dim)
        self.down1 = MamConvB_V7(28)
        self.down2 = MamConvB_V7(56)
        self.bottleneck = MamConvB_V7(112)
        self.up2 = MamConvB_V7(56)
        self.up1 = MamConvB_V7(28)


class HyMCNetL3_V7(_HyMCNetBase):
    def __init__(self, opt):
        super(HyMCNetL3_V7, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L3_V7())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)

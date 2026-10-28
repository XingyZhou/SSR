"""HyMCNetL3_V5: V4 with (1) un-shared high-frequency ConvBs and (2) SSR-style init.

Differences from model_v4.py, nothing else changed:
  1. HL, LH, HH each get their own ConvB (V4 shared one ConvB plus a band embedding).
  2. Every conv / linear inside the blocks is initialised with trunc_normal(std=0.02)
     like SSRB in Model.py, so each block starts close to identity.
  3. The FFN is Model.FFN (no internal residual). Mamba_Model.FFN adds its own
     residual, so PreNorm(FFN)(x) + x was really x + LN(x) + ffn(LN(x)); that
     extra normalised copy is what kept V4 far from identity at init.
"""
import torch
import torch.nn as nn

from Model import FFN  # no internal residual, unlike Mamba_Model.FFN
from Mamba_Model import PreNorm, Mamba_Att, ConvB, trunc_normal_, haar_dwt2d, haar_idwt2d, _HyMCNetBase
from model_v4 import WaveUNet_V4


def _ssr_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class WaveDualBlockV5(nn.Module):
    """Same-resolution block: Haar split, LF Mamba / per-band HF ConvB, IDWT, FFN."""

    def __init__(self, dim):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.lf_mamba = PreNorm(dim, Mamba_Att(dim=dim, window_size=8, shift_size=0))
        self.hf_convs = nn.ModuleList([ConvB(dim) for _ in range(3)])
        self.FFN = PreNorm(dim, FFN(dim=dim))
        self.apply(_ssr_init)
        # keep Mamba's own init for its state-space parameters (A_log, D, dt_proj); only
        # shrink the output projection so the branch starts small
        mamba = self.lf_mamba.fn.attn
        trunc_normal_(mamba.out_proj.weight, std=0.02)

    def forward(self, x):
        x = self.pos(x) + x
        ll, hl, lh, hh = haar_dwt2d(x)
        ll = self.lf_mamba(ll) + ll
        hl, lh, hh = [conv(band) for conv, band in zip(self.hf_convs, (hl, lh, hh))]
        x = haar_idwt2d(ll, hl, lh, hh)
        x = self.FFN(x) + x
        return x


class WaveUNet_V5(WaveUNet_V4):
    def __init__(self, out_dim=28):
        super().__init__(out_dim)
        for name in ("down1", "down2", "bottleneck", "up2", "up1"):
            dim = getattr(self, name).FFN.norm.normalized_shape[0]
            setattr(self, name, WaveDualBlockV5(dim))


class HyMCNetL3_V5(_HyMCNetBase):
    def __init__(self, opt):
        super(HyMCNetL3_V5, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(WaveUNet_V5())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)

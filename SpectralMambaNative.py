"""
Spectral Mamba built on the *native* (unmodified) `mamba_ssm.Mamba` block.

Same job as SpectralMamba.py -- scan along the spectral axis, one sequence per
p x p spatial patch, bidirectional -- but the SSM itself is the original
state-spaces/mamba S6 block (learned A_log, input-dependent dt/B/C, fused CUDA
kernel).  No fixed-A / sigma modification, so it is the baseline to compare
the custom frozen-sigma version against.

Interface matches SpectralMamba: x (B, C, H, W) -> (B, C, H, W), so it can be
dropped into Model.py via `from SpectralMambaNative import SpectralMamba`.
`wavelengths` is accepted for API compatibility but ignored (native Mamba has
no physical-spacing input).

Needs a CUDA GPU and the compiled `mamba_ssm` (pip install ./mamba).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from mamba_ssm import Mamba
except ImportError:  # fall back to the vendored source tree in ./mamba
    sys.path.insert(0, os.path.join(_HERE, 'mamba'))
    from mamba_ssm.modules.mamba_simple import Mamba


class SpectralMamba(nn.Module):
    """
    dim           : number of spectral bands (sequence length)
    patch         : spatial patch size p; d_model = p * p
    expand/d_state/d_conv : passed to the native Mamba block
    bidirectional : separate native Mamba for the reversed spectral order
    """

    def __init__(self, dim, patch=4, expand=2, d_state=8, d_conv=3,
                 bidirectional=True, **_ignored):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.bidirectional = bidirectional
        d_model = patch * patch
        kw = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_fwd = Mamba(**kw)
        self.mamba_bwd = Mamba(**kw) if bidirectional else None
        self.last_width = None  # kept for interface parity with SpectralMamba

    def forward(self, x, wavelengths=None):
        b, c, h_inp, w_inp = x.shape
        p = self.patch
        pad_h = (p - h_inp % p) % p
        pad_w = (p - w_inp % p) % p
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        hp, wp = x.shape[2] // p, x.shape[3] // p

        tokens = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=p, p2=p)  # (T, L, d_model)
        y = self.mamba_fwd(tokens)
        if self.bidirectional:
            y = y + self.mamba_bwd(tokens.flip(1)).flip(1)

        out = rearrange(y, '(b h w) c (p1 p2) -> b c (h p1) (w p2)', b=b, h=hp, w=wp, p1=p, p2=p)
        return out[:, :, :h_inp, :w_inp]

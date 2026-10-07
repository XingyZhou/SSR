"""
Native Mamba with the learnable decay (A) replaced by a fixed spectral width sigma.

Rationale
---------
In language, tokens differ in importance, so Mamba learns A (how long each
channel remembers) from data.  Spectral bands have no such importance ranking:
neighbouring bands are related only by distance along the spectrum.  So the
decay is not learned; it is a fixed physical prior set by one number, sigma
(in bands):

    A[d, n] = -(n + 1) / sigma          (native S4D-real init, A_log = log(n + 1), scaled by 1 / sigma)

Everything else is the unmodified state-spaces/mamba block: in_proj, causal
conv1d, input-dependent dt / B / C, D skip, SiLU gate, out_proj, and the fused
CUDA kernel.  The only structural change is that `A_log` is a frozen buffer
instead of an nn.Parameter.

SpectralMamba wraps it for (B, C, H, W) feature maps: each p x p patch is one
sequence of length C (bands), scanned in both spectral directions.

Needs a CUDA GPU and the compiled `mamba_ssm` (pip install ./mamba).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:  # fall back to the vendored source tree in ./mamba
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mamba'))
    from mamba_ssm.modules.mamba_simple import Mamba


class FixedSigmaMamba(Mamba):
    """Native Mamba whose A = -(n + 1) / sigma is fixed (never trained)."""

    def __init__(self, d_model, sigma=4.0, **kwargs):
        super().__init__(d_model, **kwargs)
        self.sigma = float(sigma)
        n = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.A_log.device)
        A_log = torch.log(n / self.sigma)[None, :].repeat(self.d_inner, 1)
        del self.A_log                                   # drop the learnable parameter
        self.register_buffer('A_log', A_log.contiguous())  # fixed; forward uses -exp(A_log)


class SpectralMamba(nn.Module):
    """
    dim           : number of spectral bands (sequence length)
    patch         : spatial patch size p; d_model = p * p
    sigma         : fixed spectral width in bands
    bidirectional : separate block for the reversed spectral order
    Other args go to the native Mamba block.  x: (B, C, H, W) -> (B, C, H, W).
    """

    def __init__(self, dim, patch=4, expand=2, d_state=8, d_conv=3, sigma=4.0,
                 bidirectional=True, **mamba_kwargs):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.sigma = float(sigma)
        self.bidirectional = bidirectional
        kw = dict(d_model=patch * patch, d_state=d_state, d_conv=d_conv, expand=expand,
                  sigma=sigma, **mamba_kwargs)
        self.mamba_fwd = FixedSigmaMamba(**kw)
        self.mamba_bwd = FixedSigmaMamba(**kw) if bidirectional else None

    def forward(self, x):
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

"""HyMCNetL3_V6: V5 with a structure-aware, Hilbert-ordered, 2D-conv Mamba on the low band.

Three encoding-level changes to the low-frequency branch of WaveDualBlockV5, no
extra scan directions, everything else (per-band HF ConvB, IDWT, FFN, SSR init)
unchanged from model_v5.py:

  1. Token = all four Haar sub-bands. cat(LL, HL, LH, HH) (4C, half res) -> 1x1 -> C.
     Each token encodes its 2x2 neighbourhood (mean, horizontal, vertical, diagonal)
     so the scan aggregates structure units, not isolated pixels. The Mamba output
     is added to LL only; the HF bands keep their own conv path.
  2. Hilbert-curve scan order instead of raster order (2D neighbours stay close in 1D).
     Falls back to a snake (boustrophedon) order when the map is not a square power of two.
  3. The causal 1D conv inside Mamba (which on a raster sequence only sees a few
     pixels to the left) is replaced by a 2D depthwise conv applied before flattening.
     The state-space parameters (A_log, D, dt_proj, x_proj) keep mamba_ssm's init.
"""
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch.cuda.amp import autocast

from Model import FFN
from Mamba_Model import PreNorm, ConvB, trunc_normal_, haar_dwt2d, haar_idwt2d, _HyMCNetBase
from model_v4 import WaveUNet_V4
from model_v5 import _ssr_init


# ----------------------------------------------------------------------------- scan order
def _hilbert_d2xy(n, d):
    x = y = 0
    s = 1
    t = d
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


@lru_cache(maxsize=None)
def scan_order(h, w):
    """Flat indices (into a row-major h*w map) in scan order, as a LongTensor on CPU."""
    if h == w and h & (h - 1) == 0:
        idx = [0] * (h * w)
        for d in range(h * w):
            x, y = _hilbert_d2xy(h, d)
            idx[d] = y * w + x
        return torch.tensor(idx)
    rows = []
    for r in range(h):
        cols = torch.arange(w) if r % 2 == 0 else torch.arange(w - 1, -1, -1)
        rows.append(r * w + cols)
    return torch.cat(rows)


# ----------------------------------------------------------------------------- Mamba variant
class Mamba2DScan(nn.Module):
    """mamba_ssm Mamba whose short conv is 2D and whose sequence follows a Hilbert curve."""

    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.m = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=expand)
        del self.m.conv1d
        self.conv2d = nn.Conv2d(self.m.d_inner, self.m.d_inner, d_conv, 1, d_conv // 2,
                                groups=self.m.d_inner, bias=True)

    @autocast(enabled=False)
    def forward(self, x):
        b, c, h, w = x.shape
        m = self.m
        xz = F.conv2d(x, m.in_proj.weight[:, :, None, None], m.in_proj.bias)  # 1x1 == Linear
        xs, z = xz.chunk(2, dim=1)
        xs = m.act(self.conv2d(xs))  # 2D neighbourhood before the scan

        order = scan_order(h, w).to(x.device)
        xs = xs.flatten(2)[:, :, order]  # b, d_inner, L in Hilbert order
        z = z.flatten(2)[:, :, order]
        L = h * w

        x_dbl = m.x_proj(rearrange(xs, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
        dt = rearrange(m.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=L)
        B = rearrange(B, "(b l) n -> b n l", l=L).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=L).contiguous()
        A = -torch.exp(m.A_log.float())
        y = selective_scan_fn(xs.contiguous(), dt.contiguous(), A, B, C, m.D.float(), z=z.contiguous(),
                              delta_bias=m.dt_proj.bias.float(), delta_softplus=True)

        out = torch.empty_like(y)
        out[:, :, order] = y  # back to raster order
        out = out.view(b, m.d_inner, h, w)
        return F.conv2d(out, m.out_proj.weight[:, :, None, None], m.out_proj.bias)


# ----------------------------------------------------------------------------- block / net
class WaveDualBlockV6(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.tok_proj = nn.Conv2d(4 * dim, dim, 1, 1, 0, bias=False)  # 2x2 structure token
        self.lf_mamba = PreNorm(dim, Mamba2DScan(dim))
        self.hf_convs = nn.ModuleList([ConvB(dim) for _ in range(3)])
        self.FFN = PreNorm(dim, FFN(dim=dim))
        self.apply(_ssr_init)
        mamba = self.lf_mamba.fn.m
        trunc_normal_(mamba.out_proj.weight, std=0.02)
        # _ssr_init touched Mamba's dt_proj / x_proj (nn.Linear); restore mamba_ssm's own init
        fresh = Mamba(d_model=dim, d_state=mamba.d_state, d_conv=4, expand=mamba.expand)
        mamba.x_proj.load_state_dict(fresh.x_proj.state_dict())
        mamba.dt_proj.load_state_dict(fresh.dt_proj.state_dict())

    def forward(self, x):
        x = self.pos(x) + x
        ll, hl, lh, hh = haar_dwt2d(x)
        tok = self.tok_proj(torch.cat([ll, hl, lh, hh], dim=1))
        ll = self.lf_mamba(tok) + ll
        hl, lh, hh = [conv(band) for conv, band in zip(self.hf_convs, (hl, lh, hh))]
        x = haar_idwt2d(ll, hl, lh, hh)
        x = self.FFN(x) + x
        return x


class WaveUNet_V6(WaveUNet_V4):
    def __init__(self, out_dim=28):
        super().__init__(out_dim)
        for name in ("down1", "down2", "bottleneck", "up2", "up1"):
            dim = getattr(self, name).FFN.norm.normalized_shape[0]
            setattr(self, name, WaveDualBlockV6(dim))


class HyMCNetL3_V6(_HyMCNetBase):
    def __init__(self, opt):
        super(HyMCNetL3_V6, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(WaveUNet_V6())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)

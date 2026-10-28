"""model_nb: band-count-agnostic versions of HyMCNetL3 / V7 / V7S for 100+ band CASSI.

The root models hard-code 28 bands (Mask_embedding, Para_Estimator, ConvB(28), the 28/56/112
U-Net widths and `g.repeat(1, 28, ...)`). This file rebuilds the same three architectures with the
band count and the U-Net widths taken from `opt`, so the identical code can be trained on CAVE (28)
or on a 100+ band cube (Chikusei 128, ICVL raw resampled, ...). Nothing in Mamba_Model.py,
model_v7.py or model_v7s.py is modified; their blocks are imported and reused.

opt fields used:
    bands      number of spectral bands nC (sequence length of the spectral scan at level 1)
    stage      unfolding stages
    size       spatial patch size
    len_shift  CASSI dispersion step (pixels per band); the 28-band code fixed this at 2
    mult       U-Net width multipliers for the 3 levels, e.g. "1,2,4" (28->28/56/112 as in HyMCNetL3)
               or "1,1,2" to keep memory down at 128 bands
    win        window side for the spectral scan (d_model = win*win), default 8
    bc_mode / sigma / dt_fixed   only for HyMCNetL3_V7SNB (FixedSigmaMamba)

Models:
    HyMCNetL3_NB      spatial raster scan (MamConvB)       -> control arm
    HyMCNetL3_V7NB    spectral scan, native S6 (MamConvB_V7)
    HyMCNetL3_V7SNB   spectral scan, FixedSigmaMamba (MamConvB_V7S)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Mamba_Model import (
    PreNorm,
    Para_Estimator,
    ConvB,
    MamConvB,
    DWT_Downsample,
    IDWT_Upsample,
)
from model_v7 import MamConvB_V7
from model_v7s import MamConvB_V7S


def _mults(opt):
    m = getattr(opt, 'mult', '1,2,4')
    if isinstance(m, str):
        m = [int(v) for v in m.split(',')]
    assert len(m) == 3, "mult needs 3 values (level1, level2, bottleneck)"
    return m


class Mask_embedding_NB(nn.Module):
    def __init__(self, nC):
        super().__init__()
        self.cnn = nn.Conv2d(nC, nC, 3, 1, 1, bias=False)
        self.mask = nn.Conv2d(nC, nC, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        return self.cnn(x) * (1 + self.mask(mask))


class MamConv_UNet_L3_NB(nn.Module):
    """MamConv_UNet_L3 with configurable band count / widths and a pluggable block.

    block_fn(dim) -> nn.Module. The Haar down/up-sampling (DWT_Downsample / IDWT_Upsample) is the
    one from Mamba_Model, so the low-frequency branch inside the downsample keeps its spatial-scan
    MamConvB exactly as in HyMCNetL3 and V7.
    """

    def __init__(self, nC, mult, block_fn):
        super().__init__()
        d1, d2, d3 = nC * mult[0], nC * mult[1], nC * mult[2]
        self.mask_embedding = Mask_embedding_NB(nC)
        self.inproj = nn.Identity() if d1 == nC else nn.Conv2d(nC, d1, 1, 1, 0, bias=False)
        self.down1 = block_fn(d1)
        self.downsample1 = DWT_Downsample(in_dim=d1, out_dim=d2)
        self.down2 = block_fn(d2)
        self.downsample2 = DWT_Downsample(in_dim=d2, out_dim=d3)
        self.bottleneck = block_fn(d3)
        self.upsample2 = IDWT_Upsample(in_dim=d3, out_dim=d2)
        self.fusion2 = nn.Conv2d(2 * d2, d2, 1, 1, 0, bias=False)
        self.up2 = block_fn(d2)
        self.upsample1 = IDWT_Upsample(in_dim=d2, out_dim=d1)
        self.fusion1 = nn.Conv2d(2 * d1, d1, 1, 1, 0, bias=False)
        self.up1 = block_fn(d1)
        self.out = nn.Conv2d(d1, nC, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        xin = x
        x0 = self.inproj(self.mask_embedding(x, mask))
        x1 = self.down1(x0)
        x = self.downsample1(x1)
        x2 = self.down2(x)
        x = self.downsample2(x2)
        x = self.bottleneck(x)
        x = self.upsample2(x)
        x = self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.upsample1(x)
        x = self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        return self.out(x) + xin


class _HyMCNetBase_NB(nn.Module):
    """_HyMCNetBase with nC / len_shift taken from opt instead of 28 / 2."""

    def __init__(self, opt):
        super().__init__()
        self.stage = opt.stage
        self.nC = opt.bands
        self.size = opt.size
        self.len_shift = getattr(opt, 'len_shift', 2)
        self.initial = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)
        self.rhos = nn.ModuleList([Para_Estimator(in_nc=self.nC) for _ in range(opt.stage)])

    def shift_back(self, x):
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=-self.len_shift * i, dims=2)
        return x[:, :, :, : self.size]

    def shift(self, x):
        x = F.pad(x, [0, (self.nC - 1) * self.len_shift, 0, 0], mode="constant", value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=self.len_shift * i, dims=2)
        return x

    def mul_PhiTg(self, Phi_shift, g):
        PhiTg = g.repeat(1, Phi_shift.shape[1], 1, 1) * Phi_shift
        return self.shift_back(PhiTg)

    def mul_Phif(self, Phi_shift, z):
        Phiz = Phi_shift * self.shift(z)
        return torch.sum(Phiz, 1).unsqueeze(1)

    def forward(self, g, input_mask=None):
        Phi, PhiPhiT = input_mask
        Phi_shift = self.shift(Phi)
        g_normal = g / self.nC * 2
        f0 = self.shift_back(g_normal.repeat(1, self.nC, 1, 1))
        f = self.initial(torch.cat([f0, Phi], dim=1))
        out = []
        for i in range(self.stage):
            rho = self.rhos[i](f)
            Phi_f = self.mul_Phif(Phi_shift, f)
            z = f + rho * self.mul_PhiTg(Phi_shift, torch.div(g - Phi_f, PhiPhiT))
            f = self.net_stage[2 * i](z, Phi)
            f = self.net_stage[2 * i + 1](f)
            out.append(f)
        return out


def _build(opt, block_fn):
    mult = _mults(opt)
    layers = []
    for _ in range(opt.stage):
        layers.append(MamConv_UNet_L3_NB(opt.bands, mult, block_fn))
        layers.append(ConvB(opt.bands))
    return nn.ModuleList(layers)


class HyMCNetL3_NB(_HyMCNetBase_NB):
    """Spatial raster-scan Mamba (HyMCNetL3) at any band count: the control arm."""

    def __init__(self, opt):
        super().__init__(opt)
        self.net_stage = _build(opt, lambda d: MamConvB(d))


class HyMCNetL3_V7NB(_HyMCNetBase_NB):
    """Spectral-axis scan with native S6 (model_v7) at any band count."""

    def __init__(self, opt):
        super().__init__(opt)
        win = getattr(opt, 'win', 8)
        self.net_stage = _build(opt, lambda d: MamConvB_V7(d, win=win))


class HyMCNetL3_V7SNB(_HyMCNetBase_NB):
    """Spectral-axis scan with FixedSigmaMamba (model_v7s) at any band count."""

    def __init__(self, opt):
        super().__init__(opt)
        win = getattr(opt, 'win', 8)
        fs = dict(sigma=getattr(opt, 'sigma', 4.0), dt_fixed=getattr(opt, 'dt_fixed', 1.0),
                  bc_mode=getattr(opt, 'bc_mode', 'selective'))
        self.net_stage = _build(opt, lambda d: MamConvB_V7S(d, win=win, **fs))

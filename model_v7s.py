"""HyMCNetL3_V7S: V7 (spectral-axis scan) with the time-axis selectivity replaced by a fixed spectral prior.

Single change relative to model_v7.py: the native `Mamba` inside SpecMamba_Att becomes
`FixedSigmaMamba` (SpectralMambaNative/SpectralMambaNative.py). Window layout (8x8 -> d_model 64),
d_state, d_conv, expand, backbone, Haar sampling, ConvB, FFN, losses and recipe are identical to V7,
so V7 vs V7S isolates "learned A / input-dependent dt" vs "fixed physical prior".

FixedSigmaMamba:
    A      fixed buffer, A[d, n] = -(n+1)/sigma      (decay scale = spectral-width prior, in bands)
    dt     fixed = dt_fixed (one band spacing), dt_proj frozen
    B, C   bc_mode  'selective' : input-dependent (native)   -> semi-selective SSM
                    's4'        : learnable per channel, content-free -> physical S4 (LTI)
                    'const'     : B = C = 1 -> pure fixed exponential filtering (lower bound)
The native V7 (learnable A, input-dependent dt) is the fourth arm of the ablation.

Caveat shared with V7: channels are physical bands only at the first level (dim 28); at dim 56 / 112
the "spectral" order is a learned channel order, so sigma there is a smoothness prior over channels.
"""
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpectralMambaNative"))
from SpectralMambaNative import FixedSigmaMamba  # noqa: E402

from Mamba_Model import PreNorm, FFN, ConvB, MamConv_UNet_L3, _HyMCNetBase  # noqa: E402
from model_v7 import SpecMamba_Att  # noqa: E402


class SpecMamba_Att_FS(SpecMamba_Att):
    """SpecMamba_Att with FixedSigmaMamba in place of the native Mamba; forward is inherited."""

    def __init__(self, dim=28, win=8, d_state=16, d_conv=4, expand=2,
                 sigma=4.0, dt_fixed=1.0, bc_mode='selective'):
        nn.Module.__init__(self)
        self.dim = dim
        self.win = win
        self.attn = FixedSigmaMamba(d_model=win * win, d_state=d_state, d_conv=d_conv, expand=expand,
                                    sigma=sigma, dt_fixed=dt_fixed, bc_mode=bc_mode)


class MambaB_V7S(nn.Module):
    def __init__(self, dim, win=8, **fs):
        super().__init__()
        self.Mamba = PreNorm(dim, SpecMamba_Att_FS(dim=dim, win=win, **fs))
        self.FFN = PreNorm(dim, FFN(dim=dim))

    def forward(self, x):
        x = self.Mamba(x) + x
        x = self.FFN(x) + x
        return x


class MamConvB_V7S(nn.Module):
    """Same layout as MamConvB / MamConvB_V7: 5x5 pos conv, MambaB, ConvB."""

    def __init__(self, dim, win=8, **fs):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.MambaB = MambaB_V7S(dim, win, **fs)
        self.ConvB = ConvB(dim)

    def forward(self, x):
        x = self.pos(x) + x
        x = self.MambaB(x)
        x = self.ConvB(x)
        return x


class MamConv_UNet_L3_V7S(MamConv_UNet_L3):
    def __init__(self, out_dim=28, **fs):
        super().__init__(out_dim)
        self.down1 = MamConvB_V7S(28, **fs)
        self.down2 = MamConvB_V7S(56, **fs)
        self.bottleneck = MamConvB_V7S(112, **fs)
        self.up2 = MamConvB_V7S(56, **fs)
        self.up1 = MamConvB_V7S(28, **fs)


class HyMCNetL3_V7S(_HyMCNetBase):
    """opt.bc_mode in {'selective', 's4', 'const'}, opt.sigma, opt.dt_fixed (defaults: selective, 4.0, 1.0)."""

    def __init__(self, opt):
        super(HyMCNetL3_V7S, self).__init__(opt)
        fs = dict(bc_mode=getattr(opt, 'bc_mode', 'selective'),
                  sigma=getattr(opt, 'sigma', 4.0),
                  dt_fixed=getattr(opt, 'dt_fixed', 1.0))
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L3_V7S(**fs))
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)

"""HyMCNetL3_Base: SSR's U-Net skeleton with plain MamConvB blocks.

Control experiment. HyMCNetL3 (Mamba_Model.py) uses MamConvB blocks but wavelet
down/up-sampling; V4/V5 use SSR's conv down/up-sampling but wavelet blocks. This
model uses SSR's skeleton AND MamConvB blocks, so it isolates the effect of the
block design from the effect of the sampling layers.
"""
import torch.nn as nn

from Mamba_Model import MamConvB, ConvB, _HyMCNetBase
from model_v4 import WaveUNet_V4


class MamConvUNet_SSRSkel(WaveUNet_V4):
    def __init__(self, out_dim=28):
        super().__init__(out_dim)
        for name in ("down1", "down2", "bottleneck", "up2", "up1"):
            dim = getattr(self, name).FFN.norm.normalized_shape[0]
            setattr(self, name, MamConvB(dim))


class HyMCNetL3_Base(_HyMCNetBase):
    def __init__(self, opt):
        super(HyMCNetL3_Base, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConvUNet_SSRSkel())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)

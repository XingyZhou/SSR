"""HyMCNetL3_V4: SSR's U-Net skeleton with wavelet dual-branch blocks.

The skeleton is identical to SSRU in Model.py: five same-resolution blocks
(down1, down2, bottleneck, up2, up1), 4x4 stride-2 conv downsampling,
ConvTranspose2d upsampling, concat skip + 1x1 fusion, global residual.
Only the block interior differs. Every block re-splits its input with a Haar DWT,
so the decoder blocks also split the FFN-mixed features again:

    x -> pos conv (+x) -> Haar split
         LL        -> LN -> Mamba      (+LL)      low frequency, half resolution
         HL,LH,HH  -> shared gated conv block      high frequency, input resolution
         Haar IDWT -> LN -> FFN (+x)

Both branches and the block itself keep an identity path.
Nothing in Mamba_Model.py or Model.py is modified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Mamba_Model import (
    GELU,
    PreNorm,
    Mamba_Att,
    FFN,
    ConvB,
    Mask_embedding,
    _HyMCNetBase,
    haar_dwt2d,
    haar_idwt2d,
)


class WaveDualBlock(nn.Module):
    """Same-resolution block: Haar split, LF Mamba / HF conv, IDWT, FFN."""

    def __init__(self, dim):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.lf_mamba = PreNorm(dim, Mamba_Att(dim=dim, window_size=8, shift_size=0))
        # one ConvB shared by the three high bands, told apart by a learned band embedding
        self.hf_conv = ConvB(dim)
        self.band_emb = nn.Parameter(torch.zeros(3, dim, 1, 1))
        self.FFN = PreNorm(dim, FFN(dim=dim))

    def forward(self, x):
        x = self.pos(x) + x
        ll, hl, lh, hh = haar_dwt2d(x)
        ll = self.lf_mamba(ll) + ll
        b = x.shape[0]
        hf = torch.stack([hl, lh, hh], dim=1) + self.band_emb  # b,3,c,h,w
        hf = self.hf_conv(hf.flatten(0, 1))  # ConvB has its own internal residuals
        hl, lh, hh = hf.view(b, 3, *hf.shape[1:]).unbind(dim=1)
        x = haar_idwt2d(ll, hl, lh, hh)  # both bands carry their identity, so IDWT already restores x
        x = self.FFN(x) + x
        return x


class WaveUNet_V4(nn.Module):
    """SSRU skeleton (Model.py) with WaveDualBlock in place of SSRB."""

    def __init__(self, out_dim=28):
        super().__init__()
        self.mask_embedding = Mask_embedding()
        self.down1 = WaveDualBlock(28)
        self.downsample1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)
        self.down2 = WaveDualBlock(56)
        self.downsample2 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)
        self.bottleneck = WaveDualBlock(112)
        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = WaveDualBlock(56)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = WaveDualBlock(28)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16  # 3 levels, and every block needs an even size
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x_in = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")
        mask = F.pad(mask, [0, pad_w, 0, pad_h], mode="reflect")

        x = self.mask_embedding(x_in, mask)
        x1 = self.down1(x)
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
        out = self.out(x) + x_in
        return out[:, :, :h_inp, :w_inp]


class HyMCNetL3_V4(_HyMCNetBase):
    """Unfolding network: per stage one WaveUNet_V4 followed by ConvB(28)."""

    def __init__(self, opt):
        super(HyMCNetL3_V4, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(WaveUNet_V4())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)

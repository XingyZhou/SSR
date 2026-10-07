"""
Native Mamba with the *time-axis* selectivity replaced by a fixed spectral
prior, and a switch for the *content-axis* selectivity (B, C).

Rationale
---------
In language, tokens differ in importance, so Mamba learns A (how long each
channel remembers) and makes dt / B / C depend on the input.  Spectral bands
have no such importance ranking: neighbouring bands are related only by their
distance along the spectrum.  So the two quantities that define the *distance
kernel* are fixed physical priors:

    A[d, n] = -(n + 1) / sigma      sigma = spectral correlation width (bands)
    dt      = dt_fixed              one band spacing, same at every position

so the decay between bands is exp(-(n + 1) * dt / sigma) whatever the content.
That is the continuity prior the SSM carries in its recurrent state.

What remains to be decided is whether the model may still choose *what* to
write into / read out of that state (B, C).  `bc_mode` selects this:

    'selective' : B_t, C_t from x_proj(x_t) as in native Mamba (S6).  The
                  distance kernel is fixed but read/write is content-dependent
                  -> "semi-selective" SSM.
    's4'        : B, C learnable per channel, independent of position and
                  content (S4-style LTI).  The model learns how to combine the
                  N kernels of widths sigma, sigma/2, ..., sigma/N.
    'const'     : B = C = 1.  Pure fixed exponential-kernel filter; lower
                  bound ablation.

Everything else is the unmodified state-spaces/mamba block: in_proj, causal
conv1d, D skip, SiLU gate, out_proj, and the fused CUDA kernel (which accepts
a non-variable (d_inner, d_state) B / C directly).

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
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
except ImportError:  # fall back to the vendored source tree in ./mamba
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mamba'))
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class FixedSigmaMamba(Mamba):
    """
    Native Mamba with A = -(n + 1) / sigma and dt = dt_fixed frozen.
    bc_mode in {'selective', 's4', 'const'} controls B / C (see module doc).
    """

    def __init__(self, d_model, sigma=4.0, dt_fixed=1.0, bc_mode='selective', **kwargs):
        super().__init__(d_model, **kwargs)
        assert bc_mode in ('selective', 's4', 'const')
        self.sigma = float(sigma)
        self.dt_fixed = float(dt_fixed)
        self.bc_mode = bc_mode

        # --- time axis: fixed distance kernel ---------------------------------
        with torch.no_grad():  # constant step: no content dependence
            self.dt_proj.weight.zero_()
            self.dt_proj.bias.fill_(self.dt_fixed + math.log(-math.expm1(-self.dt_fixed)))  # softplus^-1
        self.dt_proj.weight.requires_grad_(False)
        self.dt_proj.bias.requires_grad_(False)
        n = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.A_log.device)
        A_log = torch.log(n / self.sigma)[None, :].repeat(self.d_inner, 1)
        del self.A_log                                   # drop the learnable parameter
        self.register_buffer('A_log', A_log.contiguous())  # fixed; forward uses -exp(A_log)

        # --- content axis: B / C --------------------------------------------
        if bc_mode == 's4':
            # per-channel, position-independent (d_inner, d_state); S4D-style init
            self.B_fixed = nn.Parameter(torch.ones(self.d_inner, self.d_state))
            self.C_fixed = nn.Parameter(torch.randn(self.d_inner, self.d_state) * self.d_state ** -0.5)
        elif bc_mode == 'const':
            self.register_buffer('B_fixed', torch.ones(self.d_inner, self.d_state))
            self.register_buffer('C_fixed', torch.ones(self.d_inner, self.d_state))
        if bc_mode != 'selective':
            # x_proj's B / C columns are unused; dt columns are already dead (weight 0)
            self.x_proj.weight.requires_grad_(False)

    def forward(self, hidden_states, inference_params=None):
        if self.bc_mode == 'selective':
            return super().forward(hidden_states, inference_params)
        assert inference_params is None, "inference cache not supported for fixed B/C"
        batch, seqlen, _ = hidden_states.shape

        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                       "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        B, C = self.B_fixed.float(), self.C_fixed.float()  # (d_inner, d_state), non-variable
        if self.use_fast_path and causal_conv1d_fn is not None:
            return mamba_inner_fn(
                xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight,
                self.dt_proj.weight, self.out_proj.weight, self.out_proj.bias,
                A, B, C, self.D.float(),
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])
        # dt_proj.weight is zero, so dt = softplus(bias) = dt_fixed everywhere
        dt = xz.new_zeros(batch, self.d_inner, seqlen)
        y = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=z,
                              delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(rearrange(y, "b d l -> b l d"))


class SpectralMamba(nn.Module):
    """
    dim           : number of spectral bands (sequence length)
    patch         : spatial patch size p; d_model = p * p
    sigma         : fixed spectral width in bands
    dt_fixed      : fixed SSM step in bands (one band spacing)
    bc_mode       : 'selective' (S6 B/C) | 's4' (learnable, position-free) | 'const' (B=C=1)
    bidirectional : separate block for the reversed spectral order
    Other args go to the native Mamba block.  x: (B, C, H, W) -> (B, C, H, W).
    """

    def __init__(self, dim, patch=4, expand=2, d_state=8, d_conv=3, sigma=4.0, dt_fixed=1.0,
                 bc_mode='selective', bidirectional=True, **mamba_kwargs):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.sigma = float(sigma)
        self.bc_mode = bc_mode
        self.bidirectional = bidirectional
        kw = dict(d_model=patch * patch, d_state=d_state, d_conv=d_conv, expand=expand,
                  sigma=sigma, dt_fixed=dt_fixed, bc_mode=bc_mode, **mamba_kwargs)
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

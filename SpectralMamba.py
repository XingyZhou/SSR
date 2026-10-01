"""
Spectral Mamba with a *fixed* state matrix A whose values are determined by a
learned spectral width.

Motivation
----------
In the standard S6 (Mamba) block the state transition matrix A in R^{D x N} is a
free parameter (stored as A_log).  When the scan runs along the spectral axis
(one token per band, L = number of bands), the decay exp(dt * A) controls how
far information from neighbouring bands propagates, i.e. the *spectral
correlation width*.  Instead of learning A freely we fix its structure to

    A[d, n] = -(n + 1) / sigma_d ,      n = 0 .. N-1

where sigma_d (in units of bands) is the spectral correlation width.  sigma
can be a frozen hyperparameter (`width_mode='fixed'`, not trained), a
learnable parameter (`width_mode='param'`), or predicted per patch
(`width_mode='estimate'`).  The recommended prior is `width_mode='fixed'`
together with `selective_dt=False` so the SSM step is one band spacing and
the decay exp(-(n+1) * dt / sigma) is set entirely by that physical width.

Implementation note
-------------------
Because exp(dt * A / sigma) == exp((dt / sigma) * A_base) with
A_base = -(n + 1), a width-parameterised A is equivalent to the canonical
S4D-real initialisation with the step size rescaled by 1 / sigma.  We keep the
input scaling dt * B * x (ZOH, as in Mamba) so the only effect of sigma is on
the decay.  The same identity lets us reuse the CUDA kernel of `mamba_ssm`
(if installed) by passing u = sigma * x, delta = dt / sigma and A = A_base.
A pure-PyTorch sequential scan is used otherwise (L = 28 steps, so it is cheap).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:  # optional CUDA acceleration
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # pragma: no cover - mamba_ssm is optional
    selective_scan_fn = None


def selective_scan_ref(u, delta, A, B, C, sigma):
    """
    Pure PyTorch selective scan with width-scaled, fixed A.

    u, delta : (T, D, L)   input and step size (step size in band units)
    A        : (D, N)      fixed base matrix, A[d, n] = -(n + 1)
    B, C     : (T, N, L)   input-dependent projections
    sigma    : (T, D) or (T, 1) spectral width (bands)
    returns  : (T, D, L)
    """
    rate = delta / sigma.unsqueeze(-1)                               # (T, D, L)
    dA = torch.exp(rate.unsqueeze(2) * A[None, :, :, None])          # (T, D, N, L)
    dBu = delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)       # (T, D, N, L)
    h = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for t in range(u.shape[-1]):
        h = dA[..., t] * h + dBu[..., t]
        ys.append(torch.einsum('tdn,tn->td', h, C[..., t]))
    return torch.stack(ys, dim=-1)


class FixedASSM(nn.Module):
    """
    One scan direction of the selective SSM.  A is a buffer (not a parameter);
    its effective value -(n + 1) / sigma is given by the width passed to forward.
    dt, B, C stay input-dependent as in Mamba.  dt is initialised in band units
    (around 1 band) because the spectral sequence is short (L = 28).
    """

    def __init__(self, d_inner, d_state=8, dt_rank=4, dt_min=0.5, dt_max=2.0,
                 dt_init_floor=1e-4, fixed_A=True, selective_dt=True, A_mode='harmonic',
                 dt_fixed=1.0):
        super().__init__()
        assert A_mode in ('harmonic', 'uniform')
        self.d_inner = d_inner
        self.d_state = d_state
        self.fixed_A = fixed_A
        self.selective_dt = selective_dt
        # Physical step: one token = dt_fixed bands (default 1).  Used when
        # selective_dt=False so the decay is exp(-(n+1)*dt_fixed/sigma).
        self.dt_fixed = float(dt_fixed)
        self.dt_rank = dt_rank if selective_dt else 0

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        if selective_dt:
            self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
            dt_init_std = dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                           + math.log(dt_min)).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)

        # 'harmonic': rates (n+1)/sigma -> bank of kernels with widths sigma .. sigma/N
        # 'uniform' : rate 1/sigma for every state -> sigma is the only spectral scale
        if A_mode == 'harmonic':
            A = -torch.arange(1, d_state + 1, dtype=torch.float32)
        else:
            A = -torch.ones(d_state)
        if fixed_A:
            self.register_buffer('A', A[None, :].repeat(d_inner, 1))  # (D, N), fixed
        else:  # ablation: standard Mamba parameterisation, sigma is ignored
            self.A_log = nn.Parameter(torch.log(-A)[None, :].repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))

    def get_A(self):
        return self.A if self.fixed_A else -torch.exp(self.A_log)

    def forward(self, x, sigma):
        """
        x     : (T, D, L)
        sigma : (T, D) or (T, 1)
        """
        T, D, L = x.shape
        x_dbl = self.x_proj(rearrange(x, 't d l -> t l d'))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        if self.selective_dt:
            dt = F.softplus(self.dt_proj(dt))                        # (T, L, D)
            dt = rearrange(dt, 't l d -> t d l').contiguous()
        else:
            dt = x.new_full(x.shape, self.dt_fixed)                  # physical band spacing
        B = rearrange(B, 't l n -> t n l').contiguous()
        C = rearrange(C, 't l n -> t n l').contiguous()

        A = self.get_A()
        if not self.fixed_A:
            sigma = torch.ones(1, 1, device=x.device)
        if selective_scan_fn is not None and x.is_cuda:
            s = sigma.unsqueeze(-1).to(x.dtype)
            y = selective_scan_fn((x * s).contiguous(), (dt / s).contiguous(), A.contiguous(),
                                  B, C, D=None, z=None, delta_bias=None, delta_softplus=False)
        else:
            y = selective_scan_ref(x, dt, A, B, C, sigma.to(x.dtype))
        return y + x * self.D[None, :, None]


class SpectralMamba(nn.Module):
    """
    Bidirectional Mamba block that scans along the channel (spectral) axis.

    x: (B, C, H, W) -> (B, C, H, W).  Each p x p spatial patch becomes one
    sequence of length C whose tokens are p*p-dimensional (the patch pixels of
    that band).  The state matrix A is fixed and its decay is set by a spectral
    width sigma that is either a learnable parameter or estimated per patch.

    Args
    ----
    dim         : number of channels (= spectral bands, the sequence length)
    patch       : spatial patch size p; d_model = p * p
    expand      : d_inner = expand * d_model
    d_state     : number of SSM states N (number of exponential kernels)
    width_mode  : 'fixed'    -> sigma is a frozen hyperparameter (not trained)
                  'param'    -> one learnable sigma per inner channel
                  'estimate' -> sigma predicted per patch from the input
    sigma_min / sigma_max : bounds of sigma in bands (sigma_max defaults to dim)
    sigma_init  : width in bands (the frozen value when width_mode='fixed')
    dt_fixed    : SSM step in bands when selective_dt=False (physical token spacing)
    bidirectional : scan both directions (spectral correlation is symmetric)
    fixed_A     : False -> ablation with the standard learnable A_log (width unused)
    selective_dt: False -> step fixed to dt_fixed, so sigma alone sets the decay
    A_mode      : 'harmonic' A[d,n] = -(n+1)/sigma  or  'uniform' A[d,n] = -1/sigma
    est_hidden  : hidden width of the spectral-width estimator
    """

    def __init__(self, dim, patch=4, expand=2, d_state=8, d_conv=3,
                 width_mode='fixed', sigma_min=0.5, sigma_max=None, sigma_init=4.0,
                 bidirectional=True, fixed_A=True, selective_dt=False, A_mode='harmonic',
                 est_hidden=32, dt_fixed=1.0):
        super().__init__()
        assert width_mode in ('fixed', 'estimate', 'param')
        self.dim = dim
        self.patch = patch
        self.width_mode = width_mode
        self.bidirectional = bidirectional
        self.fixed_A = fixed_A
        self.selective_dt = selective_dt
        self.A_mode = A_mode
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(dim if sigma_max is None else sigma_max)
        if width_mode != 'fixed':
            assert self.sigma_min < sigma_init < self.sigma_max

        d_model = patch * patch
        d_inner = expand * d_model
        dt_rank = max(1, math.ceil(d_model / 4))
        self.d_inner = d_inner

        # bias=True matters: without it an all-zero band token gives z = 0 and the
        # SiLU gate silences the output at that band even though the SSM state
        # carries information from neighbouring bands.
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv // 2,
                                groups=d_inner, bias=True)
        ssm_args = dict(d_state=d_state, dt_rank=dt_rank, fixed_A=fixed_A, selective_dt=selective_dt,
                        A_mode=A_mode, dt_fixed=dt_fixed)
        self.ssm_fwd = FixedASSM(d_inner, **ssm_args)
        self.ssm_bwd = FixedASSM(d_inner, **ssm_args) if bidirectional else None
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # sigma = sigma_min + (sigma_max - sigma_min) * sigmoid(raw); solve raw for sigma_init
        if width_mode == 'fixed':
            self.register_buffer('sigma_const', torch.tensor(self.sigma_init))
        elif width_mode == 'param':
            init_raw = math.log((sigma_init - self.sigma_min) / (self.sigma_max - sigma_init))
            self.width_raw = nn.Parameter(torch.full((d_inner,), init_raw))
        else:
            init_raw = math.log((sigma_init - self.sigma_min) / (self.sigma_max - sigma_init))
            # Bands must be mixed *before* a nonlinearity, otherwise the estimator is a
            # linear function of the spectrum and cannot measure its smoothness.
            self.width_est = nn.Sequential(
                nn.Conv2d(dim, est_hidden, 1, 1, 0, bias=True),
                nn.GELU(),
                nn.Conv2d(est_hidden, est_hidden, 3, 1, 1, groups=est_hidden, bias=True),
                nn.GELU(),
                nn.Conv2d(est_hidden, 1, 1, 1, 0, bias=True),
            )
            nn.init.trunc_normal_(self.width_est[4].weight, std=.02)
            nn.init.constant_(self.width_est[4].bias, init_raw)

        self.last_width = None  # (B, 1, H/p, W/p) of the last forward, for inspection
        # in_proj / out_proj keep the default nn.Linear init (as in Mamba). A small
        # (std 0.02) init makes the output a product of three near-zero factors
        # (x_in, silu(z), W_out) and training stalls at that saddle.

    def _sigma(self, x, hp, wp):
        if self.width_mode == 'fixed':
            sigma = self.sigma_const.to(dtype=x.dtype, device=x.device).view(1, 1)
            self.last_width = sigma.detach()
            return sigma
        if self.width_mode == 'param':
            raw = self.width_raw[None, :]                                     # (1, D)
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)
            self.last_width = sigma.detach()
            return sigma
        raw = self.width_est(x)                                               # (B, 1, H, W)
        raw = F.adaptive_avg_pool2d(raw, (hp, wp))                            # one width per patch
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)
        self.last_width = sigma.detach()
        return rearrange(sigma, 'b 1 h w -> (b h w) 1')                       # (T, 1)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        p = self.patch
        pad_h = (p - h_inp % p) % p
        pad_w = (p - w_inp % p) % p
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        hp, wp = x.shape[2] // p, x.shape[3] // p

        sigma = self._sigma(x, hp, wp)

        tokens = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=p, p2=p)  # (T, L, d_model)
        xz = self.in_proj(tokens)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = rearrange(x_in, 't l d -> t d l')
        x_in = F.silu(self.conv1d(x_in))                                      # (T, D, L)

        y = self.ssm_fwd(x_in, sigma)
        if self.bidirectional:
            y = y + self.ssm_bwd(x_in.flip(-1), sigma).flip(-1)
        y = y * F.silu(rearrange(z, 't l d -> t d l'))

        out = self.out_proj(rearrange(y, 't d l -> t l d'))
        out = rearrange(out, '(b h w) c (p1 p2) -> b c (h p1) (w p2)', b=b, h=hp, w=wp, p1=p, p2=p)
        return out[:, :, :h_inp, :w_inp]

    def effective_A(self, sigma=None):
        """Return the effective state matrix -(n + 1) / sigma for inspection."""
        if sigma is None:
            sigma = self.last_width
        A = self.ssm_fwd.get_A()                                               # (D, N)
        if sigma is None or not self.fixed_A:
            return A
        if self.width_mode == 'fixed':
            return A / float(sigma.reshape(-1)[0].item() if torch.is_tensor(sigma) else sigma)
        if self.width_mode == 'param':                                        # sigma: (1, D)
            return A / sigma.reshape(-1, 1)                                    # (D, N)
        return A[None] / sigma.reshape(-1, 1, 1)                               # (T, D, N)

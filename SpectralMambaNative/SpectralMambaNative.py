"""
SpectralMambaNative: 光谱维 Mamba,基于原生 mamba_ssm 实现。

核心思想
--------
光谱与文本不同:各波段没有重要程度差异,相邻波段的关联只由光谱距离决定。
Mamba(S6)为文本设计的"时间轴选择性"(可学习 A、随输入变化的 dt)对光谱是有害的,
这里把它替换为固定的物理先验;"内容轴选择性"(B、C)是否保留作为消融对象(bc_mode)。

FixedSigmaMamba 继承原生 Mamba,只改三处:
    A   : 固定 buffer,A[d, n] = -(n+1)/sigma            (衰减尺度 = 物理光谱宽度先验)
    dt  : 固定 = dt_fixed(1 个波段间距),dt_proj 冻结    (随输入变化的 dt = 对不同波段施加不同注意力)
    B,C : 由 bc_mode 决定
            'selective' : 原生随输入变化(半选择 SSM:距离核固定,读写按内容)
            's4'        : 每通道一个可学习 (d_inner, d_state) 向量,与位置、内容无关(物理 S4,LTI)
            'const'     : B = C = 1(纯固定指数核滤波,下界对照)
其余(in_proj、causal conv1d、SiLU 门控、D 跳连、out_proj、CUDA 内核)与原生完全一致。
任意位置的波段间衰减都是 exp(-(n+1)·dt/sigma),与内容无关。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _inv_softplus(x: float) -> float:
    # softplus^{-1}(x) = x + log(1 - exp(-x))
    return x + math.log(-math.expm1(-x))


class FixedSigmaMamba(Mamba):
    """原生 Mamba,时间轴选择性(A、dt)替换为固定物理先验;B、C 由 bc_mode 决定。"""

    BC_MODES = ('selective', 's4', 'const')

    def __init__(self, d_model, sigma=4.0, dt_fixed=1.0, bc_mode='selective', **mamba_kwargs):
        if bc_mode not in self.BC_MODES:
            raise ValueError(f"bc_mode must be one of {self.BC_MODES}, got {bc_mode!r}")
        super().__init__(d_model, **mamba_kwargs)
        self.sigma = float(sigma)
        self.dt_fixed = float(dt_fixed)
        self.bc_mode = bc_mode

        # ---- A: 可学习 A_log 参数 -> 固定 buffer。父类 forward 用 A = -exp(A_log),
        #      所以这里存 log((n+1)/sigma),得到 A[d, n] = -(n+1)/sigma。
        del self.A_log
        n = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A_log = repeat(torch.log(n / self.sigma), "n -> d n", d=self.d_inner).contiguous()
        self.register_buffer("A_log", A_log)

        # ---- dt: softplus(dt_proj(x)) -> 常数 dt_fixed。weight = 0, bias = softplus^{-1}(dt_fixed),冻结。
        with torch.no_grad():
            self.dt_proj.weight.zero_()
            self.dt_proj.bias.fill_(_inv_softplus(self.dt_fixed))
        self.dt_proj.weight.requires_grad_(False)
        self.dt_proj.bias.requires_grad_(False)

        # ---- B, C
        if bc_mode == 'selective':
            pass  # 原生 x_proj 产生随输入变化的 B、C
        else:
            # x_proj 只剩 dt 的列,而 dt_proj.weight = 0,所以 x_proj 在这两档完全无效,冻结以便参数统计诚实
            self.x_proj.weight.requires_grad_(False)
            if bc_mode == 's4':
                # S4D 风格初始化:B = 1,C ~ N(0, 1/d_state)
                self.B = nn.Parameter(torch.ones(self.d_inner, self.d_state))
                self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state) * self.d_state ** -0.5)
                self.B._no_weight_decay = True
                self.C._no_weight_decay = True
            else:  # 'const'
                self.register_buffer("B", torch.ones(self.d_inner, self.d_state))
                self.register_buffer("C", torch.ones(self.d_inner, self.d_state))

    def extra_repr(self):
        return (f"d_model={self.d_model}, d_inner={self.d_inner}, d_state={self.d_state}, "
                f"sigma={self.sigma}, dt_fixed={self.dt_fixed}, bc_mode={self.bc_mode!r}")

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D) -> (B, L, D)"""
        if self.bc_mode == 'selective':
            return super().forward(hidden_states, inference_params)
        if inference_params is not None:
            raise NotImplementedError("bc_mode 's4' / 'const' 未支持推理缓存 inference_params")

        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), 固定
        B = self.B.float()                  # (d_inner, d_state), 非变量 B/C 原生内核直接支持
        C = self.C.float()

        if self.use_fast_path and causal_conv1d_fn is not None:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A, B, C,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            # dt_proj.weight == 0,dt 完全由 delta_bias 给出,直接传零
            dt = x.new_zeros((batch, self.d_inner, seqlen))
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out


class SpectralMamba(nn.Module):
    """
    (B, C, H, W) -> (B, C, H, W)
    每个 p×p 空间块作为一条长度为波段数 C 的序列,token 维度 d_model = p·p。
    双向扫描:正向和反向各一个 FixedSigmaMamba,输出相加。
    """

    def __init__(self, dim, patch=4, expand=2, d_state=8, d_conv=3,
                 sigma=4.0, dt_fixed=1.0, bc_mode='selective', bidirectional=True, **mamba_kwargs):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.bidirectional = bidirectional
        d_model = patch * patch
        kw = dict(d_model=d_model, expand=expand, d_state=d_state, d_conv=d_conv,
                  sigma=sigma, dt_fixed=dt_fixed, bc_mode=bc_mode, **mamba_kwargs)
        self.fwd = FixedSigmaMamba(**kw)
        self.bwd = FixedSigmaMamba(**kw) if bidirectional else None

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch
        pad_h, pad_w = (-h) % p, (-w) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        hp, wp = x.shape[-2:]

        seq = rearrange(x, "b c (h p1) (w p2) -> (b h w) c (p1 p2)", p1=p, p2=p)  # (B', L=C, D=p*p)
        out = self.fwd(seq)
        if self.bidirectional:
            out = out + self.bwd(seq.flip(1)).flip(1)
        out = rearrange(out, "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
                        b=b, h=hp // p, w=wp // p, p1=p, p2=p)
        if pad_h or pad_w:
            out = out[..., :h, :w]
        return out

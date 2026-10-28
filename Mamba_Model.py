import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import time
import math
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from mamba_ssm import Mamba
from torch.cuda.amp import autocast

def haar_dwt2d(x):
    """二维 Haar 离散小波变换"""
    x0 = x[:, :, 0::2, 0::2]  # 左上
    x1 = x[:, :, 1::2, 0::2]  # 左下
    x2 = x[:, :, 0::2, 1::2]  # 右上
    x3 = x[:, :, 1::2, 1::2]  # 右下
    
    ll = (x0 + x1 + x2 + x3) / 2.0
    hl = (x0 - x1 + x2 - x3) / 2.0
    lh = (x0 + x1 - x2 - x3) / 2.0
    hh = (x0 - x1 - x2 + x3) / 2.0
    return ll, hl, lh, hh

def haar_idwt2d(ll, hl, lh, hh):
    """二维 Haar 离散小波逆变换"""
    x0 = (ll + hl + lh + hh) / 2.0
    x1 = (ll - hl + lh - hh) / 2.0
    x2 = (ll + hl - lh - hh) / 2.0
    x3 = (ll - hl - lh + hh) / 2.0
    
    B, C, H_half, W_half = x0.shape
    out = torch.empty(B, C, H_half * 2, W_half * 2, device=x0.device, dtype=x0.dtype)
    out[:, :, 0::2, 0::2] = x0
    out[:, :, 1::2, 0::2] = x1
    out[:, :, 0::2, 1::2] = x2
    out[:, :, 1::2, 1::2] = x3
    return out
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.fn(x, *args, **kwargs)


class Para_Estimator(nn.Module):
    def __init__(self, in_nc=28, out_nc=1, channel=32):
        super(Para_Estimator, self).__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        # [MODIFIED]: Removed antiquated FloatTensor instantiation. Keep it simple and let device mappings pass through natively.
        self.bias = nn.Parameter(torch.ones(1))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=False),
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):

        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return x


def window_partition(x, window_size):
    # x: (B, H, W, C);  window_size (int): window size
    # Returns:     windows: (num_windows*B, window_size, window_size, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size, window_size, C);  window_size (int): Window size
    # H (int): Height of image;        W (int): Width of image
    # Returns:      x: (B, H, W, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat(
            [x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1
        )  # (1, 4, 192, 3136)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        # x is (B, H, W, C)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


#class Mamba_Att(nn.Module):
    def __init__(
        self, dim=16, window_size=16, shift_size=0, d_state=16, d_conv=3, expand=2
    ):
        super().__init__()
        self.dim = dim
        # 保留 window_size 和 shift_size 参数以兼容原有接口，但 SS2D 模式下不使用它们
        self.window_size = window_size
        self.shift_size = shift_size

        # 使用 SS2D (四相扫描) 替代原有的 Mamba
        self.attn = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=0.0,  # 如果需要 dropout 可以设值
        )

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # SS2D 期望输入为 (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # 执行四相扫描 Mamba
        x = self.attn(x)

        # 恢复形状为 (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x
class Mamba_Att(nn.Module):
    def __init__(
        self, dim=16, window_size=16, shift_size=0, d_state=16, d_conv=4, expand=2
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        # 替换为传统原生的 1D Mamba 模块
        self.attn = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    @autocast(enabled=False)
    def forward(self, x: torch.Tensor):
        # 接收输入 shape: (B, C, H, W)
        B, C, H, W = x.shape

        # 执行直接展平：(B, C, H, W) -> (B, H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        # 执行传统 1D Mamba 的因果序列扫描
        x_out = self.attn(x_flat)

        # 将序列重新折叠回 2D 空间维度：(B, H*W, C) -> (B, C, H, W)[cite: 1]
        out = x_out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out

# [DELETED]: Legacy classes Mamba_Att_old and MambaLayer were deleted here because they are absolutely dead code bloat.


# 最终 FFN 模块
class FFN(nn.Module):
    def __init__(self, dim, mult=4, drop_path=0.0):
        super().__init__()
        self.expand = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(
            dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult
        )
        self.reduce = nn.Conv2d(dim * mult, dim, 1, 1, bias=False)
        self.act = GELU()

    def forward(self, x):
        residual = x
        out = self.expand(x)
        out = self.act(out)
        out = self.dwconv(out)
        out = self.act(out)
        out = self.reduce(out)
        return residual + out  # 残差连接


class MambaB(nn.Module):
    def __init__(self, dim, window_size=16, shift_size=0):
        super().__init__()
        self.Mamba = PreNorm(
            dim, Mamba_Att(dim=dim, window_size=window_size, shift_size=0)
        )
        self.FFN = PreNorm(dim, FFN(dim=dim))

    def forward(self, x):
        x = self.Mamba(x) + x
        x = self.FFN(x) + x
        return x


class ConvB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 11, 1, 5, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm1 = nn.LayerNorm(dim)

        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(nn.Conv2d(dim, 1, 3, 1, 1, bias=False), GELU())
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False), nn.Sigmoid()
        )
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm2 = nn.LayerNorm(dim)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, mean=0.0, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x1 = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1 = self.to_out(self.to_a(x1) * self.to_v(x1)) + x

        f = self.norm2(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        out = self.out(out) + x1
        return out


class MamConvB(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=0):
        super().__init__()
        self.shift_size = shift_size
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.MambaB = MambaB(dim, window_size, shift_size=self.shift_size)
        self.ConvB = ConvB(dim)

    def forward(self, x):
        x = self.pos(x) + x
        x = self.MambaB(x)
        x = self.ConvB(x)
        return x

class DWT_Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 低频分支：使用提议的 Mamba 模块进行学习
        self.low_proj = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
        self.low_mamba = MamConvB(dim=out_dim)
        
        # 高频分支：三个高频分量拼接后输入轻量级 CNN (深度可分离卷积)
        self.high_cnn = nn.Sequential(
            nn.Conv2d(in_dim * 3, in_dim * 3, 3, 1, 1, groups=in_dim * 3, bias=False),
            GELU(),
            nn.Conv2d(in_dim * 3, out_dim, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        ll, hl, lh, hh = haar_dwt2d(x)
        
        # 处理低频 (全局结构)
        x_low = self.low_mamba(self.low_proj(ll))
        
        # 处理高频 (边缘、细节)
        x_high_cat = torch.cat([hl, lh, hh], dim=1)
        x_high = self.high_cnn(x_high_cat)
        
        # 结果融合 (这里使用残差相加降低计算量)
        return x_low + x_high

class IDWT_Upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 将深层特征投射为 4 个独立分量，用以进行逆小波变换
        self.proj = nn.Conv2d(in_dim, out_dim * 4, 1, 1, 0, bias=False)
        # 逆变换后的特征微调
        self.fusion = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        ll, hl, lh, hh = torch.chunk(x, 4, dim=1)
        out = haar_idwt2d(ll, hl, lh, hh)
        return self.fusion(out)
    
class Mask_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(28, 28, 3, 1, 1, bias=False)
        self.mask = nn.Conv2d(28, 28, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        out = self.cnn(x) * (1 + self.mask(mask))
        return out


class MamConv_UNet_L3(nn.Module):
    def __init__(self, out_dim=28):
        super(MamConv_UNet_L3, self).__init__()

        self.mask_embedding = Mask_embedding()
        self.down1 = MamConvB(dim=28)
        
        # [修改] 替换为小波下采样
        self.downsample1 = DWT_Downsample(in_dim=28, out_dim=56)
        self.down2 = MamConvB(dim=56)
        
        # [修改] 替换为小波下采样
        self.downsample2 = DWT_Downsample(in_dim=56, out_dim=112)
        self.bottleneck = MamConvB(dim=112)
        
        # [修改] 替换为小波逆变换上采样
        self.upsample2 = IDWT_Upsample(in_dim=112, out_dim=56)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = MamConvB(dim=56)
        
        # [修改] 替换为小波逆变换上采样
        self.upsample1 = IDWT_Upsample(in_dim=56, out_dim=28)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = MamConvB(dim=28)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        xin = x
        
        x0 = self.mask_embedding(x, mask)
        x1 = self.down1(x0)
        
        # 前向传播逻辑无需变动，数据流天然兼容
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
        
        out = self.out(x) + xin
        return out

class MamConv_UNet_L2(nn.Module):
    def __init__(self, out_dim=28):
        super(MamConv_UNet_L2, self).__init__()

        self.mask_embedding = Mask_embedding()
        self.down = MamConvB(dim=28)
        self.downsample = DWTDown(28, 56)
        self.bottleneck = MamConvB(dim=56)

        self.upsample = IWTUp(56, 28)
        self.fusion = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up = MamConvB(dim=28)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        xin = x
        x0 = self.mask_embedding(x, mask)
        x1 = self.down(x0)
        x = self.downsample(x1)
        x = self.bottleneck(x)
        x = self.upsample(x)
        x = self.fusion(torch.cat([x, x1], dim=1))
        x = self.up(x)
        out = self.out(x) + xin

        return out


# [NEW]: Base abstract class to handle duplicated forward and shifting logics
class _HyMCNetBase(torch.nn.Module):
    def __init__(self, opt):
        super(_HyMCNetBase, self).__init__()
        self.stage = opt.stage
        self.nC = opt.bands
        self.size = opt.size
        self.initial = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)
        para_estimator = []
        for i in range(opt.stage):
            para_estimator.append(Para_Estimator())
        self.rhos = nn.ModuleList(para_estimator)
        # Inheriting subclass MUST initialize self.net_stage in their __init__

    def shift_back(self, x, len_shift=2):
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(
                x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2
            )
        return x[:, :, :, : self.size]

    def shift(self, x, len_shift=2):
        x = F.pad(x, [0, self.nC * 2 - 2, 0, 0], mode="constant", value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    def mul_PhiTg(self, Phi_shift, g):
        # [MODIFIED]: Removed .cuda() hardcoding to allow multi-gpu/general device passover.
        temp_1 = g.repeat(1, Phi_shift.shape[1], 1, 1)
        PhiTg = temp_1 * Phi_shift
        PhiTg = self.shift_back(PhiTg)
        return PhiTg

    def mul_Phif(self, Phi_shift, z):
        z_shift = self.shift(z)
        Phiz = Phi_shift * z_shift
        Phiz = torch.sum(Phiz, 1)
        return Phiz.unsqueeze(1)

    def forward(self, g, input_mask=None):
        Phi, PhiPhiT = input_mask
        Phi_shift = self.shift(Phi, len_shift=2)
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, 28, 1, 1)
        f0 = self.shift_back(temp_g)
        f = self.initial(torch.cat([f0, Phi], dim=1))

        out = []
        for i in range(self.stage):
            # Inverse Projection
            rho = self.rhos[i](f)
            Phi_f = self.mul_Phif(Phi_shift, f)
            z = f + rho * self.mul_PhiTg(Phi_shift, torch.div(g - Phi_f, PhiPhiT))
            f = self.net_stage[2 * i](z, Phi)
            f = self.net_stage[2 * i + 1](f)
            out.append(f)
        return out


# [MODIFIED]: Subclassed from _HyMCNetBase, duplicate logic purged
class HyMCNetL3(_HyMCNetBase):
    def __init__(self, opt):
        super(HyMCNetL3, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L3())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)


# [MODIFIED]: Subclassed from _HyMCNetBase, duplicate logic purged
class HyMCNetL2(_HyMCNetBase):
    def __init__(self, opt):
        super(HyMCNetL2, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L2())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)


class LightCNN(nn.Module):
    """Depthwise 3x3 plus pointwise 1x1, the same pattern as DWT_Downsample.high_cnn."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            GELU(),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class ParallelMambaCNN(nn.Module):
    """Attention slot: a Mamba branch and a light CNN branch, fused to `dim`."""

    def __init__(self, dim, window_size=8):
        super().__init__()
        self.mamba = PreNorm(
            dim, Mamba_Att(dim=dim, window_size=window_size, shift_size=0)
        )
        self.cnn = LightCNN(dim, dim)
        self.fuse = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.mamba(x), self.cnn(x)], dim=1))


class MamConvBParallel(nn.Module):
    """MamConvB with the single Mamba attention slot replaced by ParallelMambaCNN."""

    def __init__(self, dim, window_size=8, shift_size=0):
        super().__init__()
        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.attn = ParallelMambaCNN(dim, window_size=window_size)
        self.FFN = PreNorm(dim, FFN(dim=dim))
        self.ConvB = ConvB(dim)

    def forward(self, x):
        x = self.pos(x) + x
        x = self.attn(x) + x
        x = self.FFN(x) + x
        x = self.ConvB(x)
        return x


class HFFlowMatch(nn.Module):
    """One Euler step of conditional rectified flow on the high-frequency bands.

    `bands` is cat(hl, lh, hh). The velocity field is conditioned on those bands.
    The returned map has `out_dim` channels, matching the old high_cnn.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        cond_ch = in_dim * 3
        self.v_proj = nn.Conv2d(out_dim + cond_ch + 1, out_dim, 1, 1, 0, bias=False)
        self.v_dw = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim, bias=False)
        self.act = GELU()
        self.v_out = nn.Conv2d(out_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, bands):
        b, _, h, w = bands.shape
        x = bands.new_zeros(b, self.out_dim, h, w)
        t = bands.new_zeros(b, 1, h, w)
        v = self.v_out(self.act(self.v_dw(self.v_proj(torch.cat([x, bands, t], dim=1)))))
        return x + v


class ParallelDWTDown(nn.Module):
    """Haar split, then LF Mamba and HF flow matching in parallel. No MamConvB."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.low_proj = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
        self.lf_mamba = PreNorm(
            out_dim, Mamba_Att(dim=out_dim, window_size=8, shift_size=0)
        )
        self.high_flow = HFFlowMatch(in_dim, out_dim)
        self.fuse = nn.Conv2d(out_dim * 2, out_dim, 1, 1, 0, bias=False)
        self.FFN = PreNorm(out_dim, FFN(dim=out_dim))

    def forward(self, x):
        ll, hl, lh, hh = haar_dwt2d(x)
        x_high = self.high_flow(torch.cat([hl, lh, hh], dim=1))
        x_low = self.lf_mamba(self.low_proj(ll))
        fused = self.fuse(torch.cat([x_low, x_high], dim=1))
        return self.FFN(fused) + fused


class MamConv_UNet_L3Parallel(nn.Module):
    def __init__(self, out_dim=28):
        super().__init__()
        self.mask_embedding = Mask_embedding()
        self.down1 = MamConvBParallel(dim=28)
        self.downsample1 = ParallelDWTDown(in_dim=28, out_dim=56)
        self.down2 = MamConvBParallel(dim=56)
        self.downsample2 = ParallelDWTDown(in_dim=56, out_dim=112)
        self.bottleneck = MamConvBParallel(dim=112)
        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = MamConvBParallel(dim=56)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = MamConvBParallel(dim=28)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        xin = x
        x0 = self.mask_embedding(x, mask)
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


class HyMCNetL3Parallel(_HyMCNetBase):
    """Stage-3 HyMCNet with parallel Mamba/CNN branches. Old HyMCNetL3 is unchanged."""

    def __init__(self, opt):
        super(HyMCNetL3Parallel, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L3Parallel())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)


class WaveMambaFlowBlock(nn.Module):
    """Backbone block: Haar split, LF Mamba, HF flow, Haar IDWT, then FFN.

    HFFlowMatch emits 3 * dim channels so they can be split back into hl, lh, hh
    and inverted with the processed low band. The 1x1 fuse from the parallel
    design is not used here; Haar IDWT restores the input resolution, and the
    PreNorm FFN with an outer residual is the fusion.
    """

    def __init__(self, dim):
        super().__init__()
        self.lf_mamba = PreNorm(dim, Mamba_Att(dim=dim, window_size=8, shift_size=0))
        self.high_flow = HFFlowMatch(dim, dim * 3)
        self.FFN = PreNorm(dim, FFN(dim=dim))

    def forward(self, x):
        ll, hl, lh, hh = haar_dwt2d(x)
        x_low = self.lf_mamba(ll)
        hl_o, lh_o, hh_o = torch.chunk(
            self.high_flow(torch.cat([hl, lh, hh], dim=1)), 3, dim=1
        )
        fused = haar_idwt2d(x_low, hl_o, lh_o, hh_o)
        return self.FFN(fused) + fused


class HaarChannelDown(nn.Module):
    """UNet downsample: Haar DWT, then one 1x1 conv from 4*C to out_dim."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim * 4, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        ll, hl, lh, hh = haar_dwt2d(x)
        return self.proj(torch.cat([ll, hl, lh, hh], dim=1))


class MamConv_UNet_L3_V2(nn.Module):
    def __init__(self, out_dim=28):
        super().__init__()
        self.mask_embedding = Mask_embedding()
        self.down1 = WaveMambaFlowBlock(dim=28)
        self.downsample1 = HaarChannelDown(in_dim=28, out_dim=56)
        self.down2 = WaveMambaFlowBlock(dim=56)
        self.downsample2 = HaarChannelDown(in_dim=56, out_dim=112)
        self.bottleneck = WaveMambaFlowBlock(dim=112)
        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = WaveMambaFlowBlock(dim=56)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = WaveMambaFlowBlock(dim=28)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        xin = x
        x0 = self.mask_embedding(x, mask)
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


class HyMCNetL3_V2(_HyMCNetBase):
    """Stage-3 HyMCNet whose backbone is Haar LF-Mamba / HF-flow, then IDWT and FFN."""

    def __init__(self, opt):
        super(HyMCNetL3_V2, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L3_V2())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)


class HaarSplitDown(nn.Module):
    """Haar split that returns LF and HF separately. No fusion."""

    def __init__(self, in_dim, lf_dim, hf_dim):
        super().__init__()
        self.lf_proj = nn.Conv2d(in_dim, lf_dim, 1, 1, 0, bias=False)
        self.hf_proj = nn.Conv2d(in_dim * 3, hf_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        ll, hl, lh, hh = haar_dwt2d(x)
        lf = self.lf_proj(ll)
        hf = self.hf_proj(torch.cat([hl, lh, hh], dim=1))
        return lf, hf


class DualBranchBlock(nn.Module):
    """LF Mamba and HF flow matching, fused by a 1x1 conv and a PreNorm FFN.

    Returns one tensor. The 1x1 brings the concatenated branches back to out_dim,
    and the FFN residual is the fusion.
    """

    def __init__(self, lf_dim, hf_dim, out_dim):
        super().__init__()
        if hf_dim % 3 != 0:
            raise ValueError("hf_dim must be divisible by 3 for HFFlowMatch")
        self.lf_mamba = PreNorm(lf_dim, Mamba_Att(dim=lf_dim, window_size=8, shift_size=0))
        self.high_flow = HFFlowMatch(hf_dim // 3, hf_dim)
        self.fuse = nn.Conv2d(lf_dim + hf_dim, out_dim, 1, 1, 0, bias=False)
        self.FFN = PreNorm(out_dim, FFN(dim=out_dim))

    def forward(self, lf, hf):
        fused = self.fuse(torch.cat([self.lf_mamba(lf), self.high_flow(hf)], dim=1))
        return self.FFN(fused) + fused


class MamConv_UNet_L3_V3(nn.Module):
    """Haar splits only at each downsample. Each DualBranchBlock fuses to one tensor.

    The first split consumes the full-resolution embedding and the block returns
    the half-resolution skip. The next split consumes that fused tensor. Upsamples
    are ConvTranspose2d, concatenated with the fused skip, then a 1x1 conv.
    """

    def __init__(self, out_dim=28):
        super().__init__()
        self.mask_embedding = Mask_embedding()
        self.downsample1 = HaarSplitDown(in_dim=28, lf_dim=28, hf_dim=84)
        self.block1 = DualBranchBlock(lf_dim=28, hf_dim=84, out_dim=56)
        self.downsample2 = HaarSplitDown(in_dim=56, lf_dim=56, hf_dim=168)
        self.block2 = DualBranchBlock(lf_dim=56, hf_dim=168, out_dim=112)
        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(28, out_dim, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        xin = x
        x0 = self.mask_embedding(x, mask)
        lf1, hf1 = self.downsample1(x0)
        x1 = self.block1(lf1, hf1)
        lf2, hf2 = self.downsample2(x1)
        x2 = self.block2(lf2, hf2)
        up = self.upsample2(x2)
        up = self.fusion2(torch.cat([up, x1], dim=1))
        up = self.upsample1(up)
        up = self.fusion1(torch.cat([up, x0], dim=1))
        return self.out(up) + xin


class HyMCNetL3_V3(_HyMCNetBase):
    """Stage-3 HyMCNet. Each downsample splits LF/HF; each block fuses them."""

    def __init__(self, opt):
        super(HyMCNetL3_V3, self).__init__(opt)
        netlayer = []
        for i in range(opt.stage):
            netlayer.append(MamConv_UNet_L3_V3())
            netlayer.append(ConvB(28))
        self.net_stage = nn.ModuleList(netlayer)
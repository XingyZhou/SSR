"""
GPU 冒烟测试:三档 bc_mode 的输出形状、梯度、固定先验是否生效、CUDA 快路径与参考实现一致性。
运行:python test_spectral_mamba_native.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from SpectralMambaNative import FixedSigmaMamba, SpectralMamba

torch.manual_seed(0)
dev = 'cuda'
ok = True


def check(cond, msg):
    global ok
    print(('  [PASS] ' if cond else '  [FAIL] ') + msg)
    ok &= bool(cond)


def n_params(m, trainable=True):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


SIGMA, DT = 4.0, 1.0
for mode in FixedSigmaMamba.BC_MODES:
    print(f"\n=== bc_mode = {mode!r} ===")
    m = SpectralMamba(dim=28, patch=4, sigma=SIGMA, dt_fixed=DT, bc_mode=mode).to(dev)
    x = torch.randn(2, 28, 32, 32, device=dev, requires_grad=True)
    y = m(x)
    check(y.shape == x.shape, f"shape {tuple(y.shape)}")
    check(torch.isfinite(y).all().item(), "output finite")
    y.square().mean().backward()
    check(x.grad is not None and torch.isfinite(x.grad).all().item(), "input grad finite")
    grads_ok = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.requires_grad and p.grad is not None)
    check(grads_ok, "param grads finite")

    f = m.fwd
    # 固定先验
    check(not isinstance(f.A_log, torch.nn.Parameter), "A is a buffer (not learnable)")
    A = -torch.exp(f.A_log)
    expectA = -(torch.arange(1, f.d_state + 1, device=dev) / SIGMA).expand_as(A)
    check(torch.allclose(A, expectA), f"A[d,n] = -(n+1)/sigma   (A[0,:3] = {A[0,:3].tolist()})")
    check(not f.dt_proj.weight.requires_grad and not f.dt_proj.bias.requires_grad, "dt_proj frozen")
    check(f.dt_proj.weight.abs().max().item() == 0.0, "dt_proj.weight == 0")
    check(abs(F.softplus(f.dt_proj.bias).mean().item() - DT) < 1e-5, f"softplus(dt_bias) == dt_fixed = {DT}")
    if mode == 'selective':
        check(f.x_proj.weight.requires_grad, "x_proj learnable (B, C input-dependent)")
    elif mode == 's4':
        check(isinstance(f.B, torch.nn.Parameter) and isinstance(f.C, torch.nn.Parameter), "B, C learnable (d_inner, d_state)")
        check(not f.x_proj.weight.requires_grad, "x_proj frozen (dead path)")
    else:
        check(not isinstance(f.B, torch.nn.Parameter) and (f.B == 1).all() and (f.C == 1).all(), "B = C = 1 buffers")
    print(f"  trainable params: {n_params(m):,}   (all: {n_params(m, False):,})")

    # 慢路径(torch conv + selective_scan CUDA) 与 快路径(mamba_inner_fn) 一致
    m.eval()
    with torch.no_grad():
        y_fast = m(x)
        for sub in (m.fwd, m.bwd):
            sub.use_fast_path = False
        y_slow = m(x)
        for sub in (m.fwd, m.bwd):
            sub.use_fast_path = True
    err = (y_fast - y_slow).abs().max().item()
    check(err < 1e-3, f"fast path == slow path (max abs err {err:.2e})")

    # 纯 PyTorch 参考扫描(selective_scan_ref)对照:校验固定 A/dt/B/C 确实进了内核
    if mode != 'selective':
        seq = torch.randn(3, 28, 16, device=dev)
        with torch.no_grad():
            out = f(seq)
            xz = rearrange(f.in_proj.weight @ rearrange(seq, "b l d -> d (b l)"), "d (b l) -> b d l", l=28)
            xx, z = xz.chunk(2, dim=1)
            xx = F.silu(f.conv1d(xx)[..., :28])
            dt = torch.full((3, f.d_inner, 28), DT, device=dev)
            y_ref = selective_scan_ref(xx, dt, A, f.B.float(), f.C.float(), f.D.float(), z=z)
            out_ref = f.out_proj(rearrange(y_ref, "b d l -> b l d"))
        err = (out - out_ref).abs().max().item()
        check(err < 1e-3, f"CUDA kernel == selective_scan_ref with fixed A/dt/B/C (max abs err {err:.2e})")

# 衰减核与内容无关:'const' 档单位脉冲响应应为 sum_n exp(-(n+1)*DT*k/sigma)
print("\n=== impulse response check ('const', 1-D scan core) ===")
f = FixedSigmaMamba(d_model=16, d_state=8, sigma=SIGMA, dt_fixed=DT, bc_mode='const').to(dev)
A = -torch.exp(f.A_log)
u = torch.zeros(1, f.d_inner, 28, device=dev); u[:, :, 0] = 1.0
dt = torch.full_like(u, DT)
with torch.no_grad():
    y = selective_scan_ref(u, dt, A, f.B.float(), f.C.float(), None)  # D=None, z=None
k = torch.arange(28, device=dev, dtype=torch.float32)
expect = sum(torch.exp(-(n + 1) * DT * k / SIGMA) for n in range(f.d_state))
# ZOH 离散化:deltaB_u = dt*B*u,所以首项乘 dt
err = (y[0, 0] - DT * expect).abs().max().item()
check(err < 1e-4, f"impulse response = dt * sum_n exp(-(n+1)*dt*k/sigma)  (max abs err {err:.2e})")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)

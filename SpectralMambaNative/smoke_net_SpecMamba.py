"""
整网冒烟:Model_SpecMamba.Net 在 4 档(none / selective / s4 / const)下,用真实输入尺寸跑一个 batch 的前向 + 反向,
并与 Train.py 相同的多阶段损失;统计可训练参数量与 FLOPs(fvcore 可用时)。
运行:python SpectralMambaNative/smoke_net_SpecMamba.py
"""
import os, sys, argparse, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)

import torch
from Model_SpecMamba import Net

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

torch.manual_seed(0)
B, size, bands = 2, 256, 28
g = torch.randn(B, 1, size, size + 2 * (bands - 1), device='cuda')
Phi = torch.rand(B, bands, size, size, device='cuda')
PhiPhiT = torch.rand(B, 1, size, size + 2 * (bands - 1), device='cuda') + 1.0
label = torch.rand(B, bands, size, size, device='cuda')
mse = torch.nn.MSELoss()

for mode in ['none', 'selective', 's4', 'const']:
    opt = argparse.Namespace(stage=3, bands=bands, size=size, bc_mode=mode, sigma=4.0, dt_fixed=1.0, patch=4, d_state=8)
    model = Net(opt).cuda().train()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    torch.cuda.synchronize(); t0 = time.time()
    out = model(g=g, input_mask=(Phi, PhiPhiT))
    loss = sum(w * torch.sqrt(mse(out[i], label)) for w, i in zip((1.0, 0.7, 0.5, 0.3), (2, 1, 0, -1)))
    loss.backward()
    torch.cuda.synchronize(); dt = time.time() - t0
    bad = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{mode:9s}] out {tuple(out[-1].shape)}  loss {loss.item():.4f}  fwd+bwd {dt:.2f}s  "
          f"peak mem {mem:.2f} GB  params trainable {n_tr/1e6:.4f}M / all {n_all/1e6:.4f}M  "
          f"{'grads OK' if not bad else 'BAD GRADS: ' + str(bad[:3])}")
    if FlopCountAnalysis is not None:
        model.eval()
        with torch.no_grad():
            fl = FlopCountAnalysis(model, inputs=(g[:1], (Phi[:1], PhiPhiT[:1])))
            fl.unsupported_ops_warnings(False); fl.uncalled_modules_warnings(False)
            print(f"            FLOPs (fvcore, batch 1) = {fl.total()/1e9:.2f} G  "
                  f"(注意: mamba_inner_fn 为自定义 CUDA 算子, fvcore 不计其内部 FLOPs)")
    del model, out, loss; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

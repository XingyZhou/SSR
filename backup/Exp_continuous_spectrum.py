# [备份注释] 文件名: Exp_continuous_spectrum.py
# [备份注释] 作用: 实验: 连续光谱 Mamba, 零样本传感器/波段数适配
# [备份注释] 备份来源: 提交 5da19fd 时的原文件, 内容未改动

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SpectralMamba import SpectralMamba

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='./exp_out/continuous')
parser.add_argument('--iters', default=800, type=int)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--crop', default=16, type=int)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, wavelengths=None):
        return x + self.fn(x, wavelengths=wavelengths)

def build_model(name):
    if name == 'continuous':
        return Residual(SpectralMamba(
            dim=28, patch=4, expand=2, d_state=8,
            width_mode='fixed', sigma_init=0.1,  # Normalized correlation width (30nm / 300nm = 0.1)
            selective_dt=False, A_mode='harmonic', fixed_A=True))
    elif name == 'learned_dt':
        return Residual(SpectralMamba(
            dim=28, patch=4, expand=2, d_state=8,
            width_mode='fixed', sigma_init=0.1,
            selective_dt=True, A_mode='harmonic', fixed_A=True))
    else:
        raise ValueError(name)

def generate_sensor_data(b, h, w, wavelengths, gen):
    L = len(wavelengths)
    x = torch.zeros(b, L, h, w)
    for bi in range(b):
        num_peaks = 3
        centers = torch.rand(num_peaks, generator=gen) * 300 + 400
        widths = torch.rand(num_peaks, generator=gen) * 30 + 10
        amps = torch.rand(num_peaks, generator=gen) * 0.8 + 0.2
        mixing = torch.rand(num_peaks, h, w, generator=gen)
        mixing = F.avg_pool2d(mixing.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
        mixing = mixing / mixing.sum(dim=0, keepdim=True)
        waves = wavelengths.view(L, 1, 1)
        for i in range(num_peaks):
            spectrum = amps[i] * torch.exp(-0.5 * ((waves - centers[i]) / widths[i])**2)
            x[bi] += spectrum * mixing[i]
    return x.clamp(0, 1)

def corrupt(x, gen):
    noise = torch.randn(x.shape, generator=gen) * 0.05
    return (x + noise).clamp(0, 1)

def train_model(model, source_waves, iters, tag):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-4)
    gen = torch.Generator().manual_seed(args.seed + 1)
    model.train()
    for it in range(iters):
        gt = generate_sensor_data(args.batch, args.crop, args.crop, source_waves, gen)
        inp = corrupt(gt, gen)
        # Normalize wavelengths to [0, 1] range based on typical visible spectrum (400-700nm)
        # This prevents delta from being too large (e.g. 10-50) which blows up the ODE input term (delta * B * x)
        w_norm = (source_waves - 400.0) / 300.0
        out = model(inp, wavelengths=w_norm)
        loss = F.mse_loss(out, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model

@torch.no_grad()
def evaluate(model, target_waves, gen):
    model.eval()
    gt = generate_sensor_data(16, args.crop, args.crop, target_waves, gen)
    inp = corrupt(gt, gen)
    w_norm = (target_waves - 400.0) / 300.0
    out = model(inp, wavelengths=w_norm).clamp(0, 1)
    mse = F.mse_loss(out, gt).item()
    psnr = 10 * np.log10(1.0 / mse)
    return psnr, inp, out, gt

def main():
    source_waves = torch.linspace(400, 700, steps=28)
    target_waves_1 = torch.linspace(400, 700, steps=15)
    w1 = torch.linspace(400, 500, steps=8)
    w2 = torch.linspace(550, 600, steps=5)
    w3 = torch.linspace(650, 700, steps=7)
    target_waves_2 = torch.cat([w1, w2, w3])
    
    models = ['learned_dt', 'continuous']
    results = {}
    gen_test = torch.Generator().manual_seed(999)
    
    for name in models:
        print(f"\n--- Training {name} on Source Sensor (28 bands) ---")
        torch.manual_seed(args.seed)
        model = build_model(name)
        model = train_model(model, source_waves, args.iters, name)
        
        psnr_src, _, _, _ = evaluate(model, source_waves, gen_test)
        print(f"  Source (28 uniform) PSNR : {psnr_src:.2f} dB")
        psnr_tgt1, _, _, _ = evaluate(model, target_waves_1, gen_test)
        print(f"  Target 1 (15 uniform) PSNR : {psnr_tgt1:.2f} dB")
        psnr_tgt2, _, _, _ = evaluate(model, target_waves_2, gen_test)
        print(f"  Target 2 (20 non-uniform) PSNR : {psnr_tgt2:.2f} dB")
        
        results[name] = {
            'source_28': psnr_src,
            'target_15_uniform': psnr_tgt1,
            'target_20_gaps': psnr_tgt2
        }
        
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, [results['learned_dt']['source_28'], 
                         results['learned_dt']['target_15_uniform'], 
                         results['learned_dt']['target_20_gaps']], 
           width, label='Learned dt (Discrete Tokens)')
    ax.bar(x + width/2, [results['continuous']['source_28'], 
                         results['continuous']['target_15_uniform'], 
                         results['continuous']['target_20_gaps']], 
           width, label='Continuous Mamba (Physical $\Delta\lambda$)')
    ax.set_ylabel('Denoising PSNR (dB)')
    ax.set_title('Zero-Shot Sensor Adaptation')
    ax.set_xticks(x)
    ax.set_xticklabels(['Source\n(28 uniform)', 'Target A\n(15 uniform)', 'Target B\n(20 w/ gaps)'])
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_sensor_adaptation.png'), dpi=130)
    plt.close(fig)

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
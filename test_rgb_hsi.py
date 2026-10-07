import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('./exp_out/rgb2hsi', exist_ok=True)

class ContinuousSpectralMixer(nn.Module):
    def __init__(self, d_model=64, d_state=8, sigma_init=20.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.sigma = sigma_init
        self.proj_BC = nn.Linear(d_model, 2 * d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        A = -torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.register_buffer('A_base', A)

    def forward(self, x, wavelengths):
        B, L, D = x.shape
        device = x.device
        dt = torch.zeros(L, device=device)
        if L > 1:
            dt[1:] = torch.abs(wavelengths[1:] - wavelengths[:-1])
            dt[0] = dt[1:].median()
        else:
            dt[0] = 1.0
        dt = dt.view(1, 1, L).expand(B, D, L)
        
        bc = self.proj_BC(x)
        B_proj, C_proj = torch.split(bc, [self.d_state, self.d_state], dim=-1)
        B_proj = B_proj.permute(0, 2, 1).contiguous()
        C_proj = C_proj.permute(0, 2, 1).contiguous()
        u = x.permute(0, 2, 1).contiguous()
        
        rate = dt / self.sigma
        dA = torch.exp(rate.unsqueeze(2) * self.A_base.unsqueeze(0).unsqueeze(-1))
        dBu = dt.unsqueeze(2) * B_proj.unsqueeze(1) * u.unsqueeze(2)
        
        h = torch.zeros(B, D, self.d_state, device=device)
        ys = []
        for t in range(L):
            h = dA[..., t] * h + dBu[..., t]
            y_t = torch.einsum('bdn,bn->bd', h, C_proj[..., t])
            ys.append(y_t)
            
        y = torch.stack(ys, dim=-1)
        y = y + u * self.D.unsqueeze(0).unsqueeze(-1)
        return y.permute(0, 2, 1)

class RGB2HSI(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )
        self.wave_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.mamba = ContinuousSpectralMixer(d_model=d_model, d_state=8, sigma_init=20.0)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, rgb, wavelengths):
        b, _, h, w = rgb.shape
        L = len(wavelengths)
        cond = self.rgb_encoder(rgb).permute(0, 2, 3, 1).reshape(-1, self.d_model)
        w_norm = (wavelengths - 400.0) / 300.0
        w_embed = self.wave_mlp(w_norm.unsqueeze(1))
        tokens = cond.unsqueeze(1) + w_embed.unsqueeze(0)
        
        y = self.mamba(tokens, w_norm)
        out = self.out_proj(y).squeeze(-1)
        return out.view(b, h, w, L).permute(0, 3, 1, 2)

def generate_data(wavelengths, b=4):
    L = len(wavelengths)
    waves = wavelengths.view(1, L, 1, 1)
    gt = torch.zeros(b, L, 16, 16)
    for i in range(b):
        num_peaks = 3
        centers = torch.rand(num_peaks) * 200 + 450
        widths = torch.rand(num_peaks) * 40 + 20
        amps = torch.rand(num_peaks) * 0.8 + 0.2
        for p in range(num_peaks):
            spectrum = amps[p] * torch.exp(-0.5 * ((waves - centers[p]) / widths[p])**2)
            gt[i] += spectrum[0]
    gt = gt.clamp(0, 1)
    
    rgb_centers = torch.tensor([600.0, 530.0, 450.0]).view(1, 3, 1, 1, 1)
    rgb_widths = torch.tensor([40.0, 40.0, 40.0]).view(1, 3, 1, 1, 1)
    crf = torch.exp(-0.5 * ((waves.unsqueeze(1) - rgb_centers) / rgb_widths)**2)
    crf_sum = crf.sum(dim=2)
    rgb = (gt.unsqueeze(1) * crf).sum(dim=2) / crf_sum.clamp(min=1e-6)
    return rgb, gt
    return rgb, gt

def main():
    torch.manual_seed(42)
    train_waves = torch.linspace(400, 700, 31)
    model = RGB2HSI(d_model=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    
    print("Training on 31 bands...")
    for it in range(300):
        rgb, gt = generate_data(train_waves, b=8)
        pred = model(rgb, train_waves)
        loss = F.mse_loss(pred, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it+1) % 50 == 0:
            print(f"Iter {it+1:03d}, Loss: {loss.item():.4f}")

    print("\nTesting Zero-Shot on 150 bands...")
    test_waves = torch.linspace(400, 700, 150)
    rgb_test, gt_test_31 = generate_data(train_waves, b=1)
    _, gt_test_150 = generate_data(test_waves, b=1)
    
    with torch.no_grad():
        pred_31 = model(rgb_test, train_waves)
        pred_150 = model(rgb_test, test_waves)

    fig, ax = plt.subplots(figsize=(8, 5))
    pixel_31 = pred_31[0, :, 8, 8].numpy()
    pixel_150 = pred_150[0, :, 8, 8].numpy()
    true_150 = gt_test_150[0, :, 8, 8].numpy()

    ax.plot(test_waves.numpy(), true_150, '--', color='black', alpha=0.5, label='Ground Truth (Continuous)')
    ax.plot(train_waves.numpy(), pixel_31, 'o', color='blue', markersize=6, label='Train Resolution (31 bands)')
    ax.plot(test_waves.numpy(), pixel_150, '-', color='red', lw=2, label='Zero-Shot Mamba (150 bands)')

    ax.axvspan(580, 620, color='red', alpha=0.1, label='R constraint')
    ax.axvspan(510, 550, color='green', alpha=0.1, label='G constraint')
    ax.axvspan(430, 470, color='blue', alpha=0.1, label='B constraint')

    ax.set_title("Resolution-Agnostic RGB to HSI using Continuous Mamba")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig('./exp_out/rgb2hsi/zero_shot_rgb2hsi.png', dpi=150)
    print("Saved plot to ./exp_out/rgb2hsi/zero_shot_rgb2hsi.png")

if __name__ == '__main__':
    main()

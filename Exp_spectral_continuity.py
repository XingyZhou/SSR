"""
Small-scale CPU experiment: does the fixed-A spectral Mamba learn spectral continuity?

Part A (synthetic, controlled width)
    Per-pixel spectra are sampled from a Gaussian process over the band axis with an
    RBF kernel of length-scale ell in {1, 2, 4, 8} bands (one ell per 8x8 block).
    The model is trained to recover randomly dropped / noisy bands.  If the block
    learns spectral continuity, the estimated width sigma should increase with ell.

Part B (real data, Indian Pines -> 28 broad bands)
    Same band-dropout recovery task.  Compared models (similar parameter budgets):
      ours      : SpectralMamba, fixed A, per-patch estimated width
      learnA    : same block, standard learnable A_log (ablation)
      specMLP   : 1x1-conv MLP across bands (spectral context, no continuity prior)
      spatialCNN: 3x3 conv applied to every band independently (no spectral context)
    Reports PSNR on dropped bands, the spectral influence matrix |dy_i/dx_j|, example
    spectra and the estimated width map vs. local spectral roughness.

Usage:
    python Exp_spectral_continuity.py --data /path/to/Indian_pines_corrected.mat --out_dir ./exp_out
"""
import argparse
import json
import math
import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from SpectralMamba import SpectralMamba

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='/tmp/hsi/ip.mat')
parser.add_argument('--out_dir', default='./exp_out')
parser.add_argument('--iters', default=1500, type=int)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--crop', default=32, type=int)
parser.add_argument('--bands', default=28, type=int)
parser.add_argument('--drop_p', default=0.3, type=float, help='fraction of dropped bands')
parser.add_argument('--noise', default=0.02, type=float)
parser.add_argument('--lr', default=2e-3, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--threads', default=4, type=int)
args = parser.parse_args()

torch.set_num_threads(args.threads)
os.makedirs(args.out_dir, exist_ok=True)
L = args.bands


# ----------------------------------------------------------------------------- models
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class SpectralMLP(nn.Module):
    def __init__(self, bands, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(bands, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, bands, 1))

    def forward(self, x):
        return self.net(x)


class SpatialCNN(nn.Module):
    """Shared 2-D CNN applied to each band separately: no spectral mixing at all."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(hidden, 1, 3, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return self.net(x.reshape(b * c, 1, h, w)).reshape(b, c, h, w)


def build(name):
    if name == 'ours':
        return Residual(SpectralMamba(L, width_mode='estimate'))
    if name == 'ours_param':
        return Residual(SpectralMamba(L, width_mode='param'))
    if name == 'learnA':
        return Residual(SpectralMamba(L, fixed_A=False))
    if name == 'specMLP':
        return Residual(SpectralMLP(L))
    if name == 'spatialCNN':
        return Residual(SpatialCNN())
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- task
def corrupt(x, gen):
    """Drop a random subset of bands (set to 0) and add Gaussian noise."""
    b = x.shape[0]
    drop = torch.rand(b, L, 1, 1, generator=gen) < args.drop_p
    noise = torch.randn(x.shape, generator=gen) * args.noise
    return (x + noise) * (~drop).float(), drop.expand_as(x)


def psnr(pred, gt, mask=None):
    err = (pred - gt) ** 2
    mse = err[mask].mean() if mask is not None else err.mean()
    return float(10 * torch.log10(1.0 / mse))


def train(model, sampler, iters, tag):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=args.lr * 0.02)
    gen = torch.Generator().manual_seed(args.seed + 1)
    t0 = time.time()
    for it in range(iters):
        gt = sampler(args.batch)
        inp, _ = corrupt(gt, gen)
        loss = F.mse_loss(model(inp), gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == iters - 1:
            print('  [%s] iter %4d loss %.5f  (%.0fs)' % (tag, it, loss.item(), time.time() - t0), flush=True)
    return model


@torch.no_grad()
def evaluate(model, gt, inp, drop):
    model.eval()
    out = model(inp).clamp(0, 1)
    return dict(psnr_all=psnr(out, gt), psnr_dropped=psnr(out, gt, drop),
                psnr_kept=psnr(out, gt, ~drop)), out


def spectral_jacobian(model, x):
    """|dy_i / dx_j| averaged over pixels -> (L, L) spectral influence matrix."""
    model.eval()
    x = x.clone().requires_grad_(True)
    y = model(x)
    J = torch.zeros(L, L)
    for i in range(L):
        g, = torch.autograd.grad(y[:, i].sum(), x, retain_graph=i < L - 1)
        J[i] = g.abs().mean(dim=(0, 2, 3))
    return J


def band_halfwidth(J):
    """Average distance (in bands) over which an output band draws >10% of its peak influence."""
    widths = []
    for i in range(L):
        row = J[i] / J[i].max()
        idx = torch.nonzero(row > 0.1).flatten()
        widths.append(float((idx.max() - idx.min()).item()) / 2)
    return float(np.mean(widths))


# ----------------------------------------------------------------------------- Part A
def gp_cube(b, h, w, ells, gen, block=8):
    """GP spectra with RBF kernel; one length-scale per block x block region."""
    idx = torch.arange(L, dtype=torch.float32)
    chol = {}
    for ell in ells:
        K = torch.exp(-(idx[:, None] - idx[None, :]) ** 2 / (2 * ell ** 2)) + 1e-4 * torch.eye(L)
        chol[ell] = torch.linalg.cholesky(K)
    nb_h, nb_w = h // block, w // block
    ell_map = torch.tensor(ells)[torch.randint(len(ells), (b, nb_h, nb_w), generator=gen)]
    x = torch.zeros(b, L, h, w)
    for bi in range(b):
        for i in range(nb_h):
            for j in range(nb_w):
                z = torch.randn(L, block * block, generator=gen)
                s = chol[float(ell_map[bi, i, j])] @ z
                s = 0.5 + 0.15 * s  # spectra around 0.5 with std 0.15
                x[bi, :, i * block:(i + 1) * block, j * block:(j + 1) * block] = s.reshape(L, block, block)
    return x.clamp(0, 1), ell_map


def part_a():
    print('\n=== Part A: synthetic spectra with known correlation length ===')
    ells = [1.0, 2.0, 4.0, 8.0]
    gen = torch.Generator().manual_seed(args.seed)
    sampler = lambda b: gp_cube(b, args.crop, args.crop, ells, gen)[0]
    model = train(build('ours'), sampler, args.iters, 'synthetic/ours')

    gen_test = torch.Generator().manual_seed(args.seed + 7)
    gt, ell_map = gp_cube(16, args.crop, args.crop, ells, gen_test)
    inp, drop = corrupt(gt, gen_test)
    metrics, _ = evaluate(model, gt, inp, drop)
    sm = model.fn
    with torch.no_grad():
        sm(inp)
    sigma = sm.last_width[:, 0]                                            # (B, H/4, W/4)
    ell_patch = ell_map.repeat_interleave(2, 1).repeat_interleave(2, 2)    # block 8 -> patch 4
    res = {'metrics': metrics, 'sigma_by_ell': {}}
    for ell in ells:
        s = sigma[ell_patch == ell]
        res['sigma_by_ell'][str(ell)] = dict(mean=float(s.mean()), std=float(s.std()),
                                            median=float(s.median()))
        print('  true ell = %.0f bands -> estimated sigma = %.2f +- %.2f (median %.2f)' %
              (ell, s.mean(), s.std(), s.median()))
    from scipy.stats import spearmanr
    rho = spearmanr(ell_patch.flatten().numpy(), sigma.flatten().numpy()).correlation
    res['spearman_ell_sigma'] = float(rho)
    print('  Spearman(ell, sigma) = %.3f ; dropped-band PSNR = %.2f dB' % (rho, metrics['psnr_dropped']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].boxplot([sigma[ell_patch == e].numpy() for e in ells])
    ax[0].set_xticklabels([str(int(e)) for e in ells])
    ax[0].set_xlabel('true GP length-scale (bands)')
    ax[0].set_ylabel('estimated spectral width sigma (bands)')
    ax[0].set_title('learned width vs. true width  (Spearman %.2f)' % rho)
    im0 = ax[1].imshow(ell_patch[0].numpy(), cmap='viridis')
    ax[1].set_title('true length-scale (one test cube)')
    plt.colorbar(im0, ax=ax[1], fraction=0.046)
    im1 = ax[2].imshow(sigma[0].numpy(), cmap='viridis')
    ax[2].set_title('estimated sigma')
    plt.colorbar(im1, ax=ax[2], fraction=0.046)
    ax[1].axis('off')
    ax[2].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_synthetic_width.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- Part B
def load_real():
    d = sio.loadmat(args.data)
    cube = [v for k, v in d.items() if not k.startswith('__')][0].astype(np.float32)
    nb = cube.shape[2] // L
    cube = cube[:, :, :nb * L].reshape(cube.shape[0], cube.shape[1], L, nb).mean(-1)  # L broad bands
    cube = cube / np.percentile(cube, 99.9)
    cube = torch.from_numpy(cube).permute(2, 0, 1).clamp(0, 1)                    # (L, H, W)
    return cube


def part_b():
    print('\n=== Part B: real data (%s) ===' % os.path.basename(args.data))
    cube = load_real()
    _, H, W = cube.shape
    test_rows = 32
    train_cube, test_cube = cube[:, :H - test_rows], cube[:, H - test_rows:]
    gen = torch.Generator().manual_seed(args.seed)

    def sampler(b):
        out = []
        for _ in range(b):
            r = torch.randint(0, train_cube.shape[1] - args.crop + 1, (1,), generator=gen).item()
            c = torch.randint(0, train_cube.shape[2] - args.crop + 1, (1,), generator=gen).item()
            x = train_cube[:, r:r + args.crop, c:c + args.crop]
            if torch.rand(1, generator=gen) < 0.5:
                x = x.flip(-1)
            out.append(x)
        return torch.stack(out)

    gen_test = torch.Generator().manual_seed(args.seed + 7)
    gt = test_cube[None]
    inp, drop = corrupt(gt, gen_test)
    print('  input PSNR: all %.2f dB, dropped bands %.2f dB' % (psnr(inp, gt), psnr(inp, gt, drop)))

    names = ['ours', 'learnA', 'specMLP', 'spatialCNN']
    results, outputs, jac = {}, {}, {}
    for name in names:
        torch.manual_seed(args.seed)
        model = build(name)
        print('  model %-10s params %d' % (name, n_params(model)))
        model = train(model, sampler, args.iters, 'real/' + name)
        m, out = evaluate(model, gt, inp, drop)
        m['params'] = n_params(model)
        J = spectral_jacobian(model, gt[:, :, :16, :16])
        m['influence_halfwidth_bands'] = band_halfwidth(J)
        results[name], outputs[name], jac[name] = m, out[0], J
        print('  %-10s PSNR all %.2f | dropped %.2f | kept %.2f | influence half-width %.1f bands' %
              (name, m['psnr_all'], m['psnr_dropped'], m['psnr_kept'], m['influence_halfwidth_bands']))
        if name == 'ours':
            ours_model = model

    # width map vs. local spectral roughness on the full image
    sm = ours_model.fn
    with torch.no_grad():
        sm(cube[None])
        sigma_map = sm.last_width[0, 0]                                         # (H/4, W/4)
        d2 = cube[2:] - 2 * cube[1:-1] + cube[:-2]
        rough = d2.abs().mean(0) / (cube.std(0) + 1e-3)                          # (H, W)
        pad_h, pad_w = sigma_map.shape[0] * 4 - H, sigma_map.shape[1] * 4 - W
        rough = F.pad(rough[None, None], [0, pad_w, 0, pad_h], mode='reflect')
        rough_map = F.avg_pool2d(rough, 4)[0, 0]
    from scipy.stats import spearmanr
    rho = spearmanr(rough_map.flatten().numpy(), sigma_map.flatten().numpy()).correlation
    results['spearman_roughness_sigma'] = float(rho)
    results['sigma_range'] = [float(sigma_map.min()), float(sigma_map.max())]
    print('  Spearman(spectral roughness, sigma) = %.3f ; sigma range [%.2f, %.2f] bands' %
          (rho, sigma_map.min(), sigma_map.max()))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, name in zip(ax, names):
        a.imshow(jac[name].numpy(), cmap='magma')
        a.set_title('%s\n|dy_i/dx_j|, half-width %.1f bands' % (name, results[name]['influence_halfwidth_bands']))
        a.set_xlabel('input band j')
        a.set_ylabel('output band i')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_real_jacobian.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    pix = [(5, 20), (16, 70), (28, 120)]
    for a, (r, c) in zip(ax, pix):
        x = np.arange(L)
        a.plot(x, gt[0, :, r, c].numpy(), 'k-', lw=2, label='ground truth')
        kept = ~drop[0, :, r, c]
        a.plot(x[kept.numpy()], inp[0, :, r, c][kept].numpy(), 'o', color='gray', ms=4, label='input (kept bands)')
        a.plot(x[~kept.numpy()], np.zeros((~kept).sum().item()), 'x', color='red', ms=6, label='dropped bands')
        for name, st in [('ours', 'b-'), ('learnA', 'g--'), ('specMLP', 'm:'), ('spatialCNN', 'c-.')]:
            a.plot(x, outputs[name][:, r, c].numpy(), st, lw=1.3,
                   label='%s (%.1f dB)' % (name, results[name]['psnr_dropped']))
        a.set_xlabel('band')
        a.set_title('pixel (%d, %d)' % (r, c))
    ax[0].set_ylabel('reflectance')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_real_spectra.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rgb = cube[[L * 2 // 3, L // 2, L // 4]].permute(1, 2, 0).numpy()
    ax[0].imshow(np.clip(rgb / rgb.max(), 0, 1))
    ax[0].set_title('false-colour image')
    im1 = ax[1].imshow(sigma_map.numpy(), cmap='viridis')
    ax[1].set_title('estimated spectral width sigma (bands)')
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(rough_map.numpy(), cmap='viridis')
    ax[2].set_title('spectral roughness |d2 x| / std   (Spearman %.2f)' % rho)
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_real_width_map.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    return results


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    all_res = {'args': vars(args), 'synthetic': part_a(), 'real': part_b()}
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(all_res, f, indent=2)
    print('\nsaved to', args.out_dir)

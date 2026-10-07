"""
Frozen-σ spectral Mamba: Δ is the physical band spacing, σ is a hyperparameter
that is never updated from the training set.

  A[d, n] = -(n + 1) / σ ,   Δ = 1 band

The rest of the block (B, C, gates, projections) is still trained.  The
question is whether this prior is usable for spectral continuity, and whether
the chosen σ matters.

Part A  Synthetic GP spectra (known length-scale ell).
        Sweep frozen σ, then match / mismatch σ against ell.
Part B  Indian Pines (28 broad bands), same band-dropout task.

Usage:
    python Exp_spectral_continuity.py --data /tmp/hsi/ip.mat --out_dir ./exp_out
"""
import argparse
import json
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
parser.add_argument('--iters', default=800, type=int)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--crop', default=32, type=int)
parser.add_argument('--bands', default=28, type=int)
parser.add_argument('--drop_p', default=0.3, type=float)
parser.add_argument('--noise', default=0.02, type=float)
parser.add_argument('--lr', default=2e-3, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--threads', default=4, type=int)
parser.add_argument('--parts', default='ab', help="'a', 'b' or 'ab'")
parser.add_argument('--eval_n', default=32, type=int, help='images in the per-ell evaluation set (Part A)')
args = parser.parse_args()

torch.set_num_threads(args.threads)
os.makedirs(args.out_dir, exist_ok=True)
L = args.bands


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
    """Per-band 2-D CNN: no spectral mixing."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(hidden, 1, 3, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return self.net(x.reshape(b * c, 1, h, w)).reshape(b, c, h, w)


def prior(sigma, A_mode='harmonic'):
    """Frozen σ, frozen Δ = 1 band.  σ is a buffer, not a parameter."""
    return Residual(SpectralMamba(
        L, width_mode='fixed', sigma_init=float(sigma),
        selective_dt=False, dt_fixed=1.0, A_mode=A_mode, fixed_A=True))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def n_trainable_sigma(m):
    return sum(p.numel() for n, p in m.named_parameters() if 'width' in n or 'sigma' in n)


def corrupt(x, gen):
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
        if it % 200 == 0 or it == iters - 1:
            print('  [%s] iter %4d loss %.5f  (%.0fs)' % (tag, it, loss.item(), time.time() - t0), flush=True)
    return model


@torch.no_grad()
def evaluate(model, gt, inp, drop):
    model.eval()
    out = model(inp).clamp(0, 1)
    return dict(psnr_all=psnr(out, gt), psnr_dropped=psnr(out, gt, drop),
                psnr_kept=psnr(out, gt, ~drop)), out


def spectral_jacobian(model, x):
    model.eval()
    x = x.clone().requires_grad_(True)
    y = model(x)
    J = torch.zeros(L, L)
    for i in range(L):
        g, = torch.autograd.grad(y[:, i].sum(), x, retain_graph=i < L - 1)
        J[i] = g.abs().mean(dim=(0, 2, 3))
    return J


def band_halfwidth(J):
    widths = []
    for i in range(L):
        row = J[i] / (J[i].max() + 1e-12)
        idx = torch.nonzero(row > 0.1).flatten()
        if len(idx) == 0:
            widths.append(0.0)
        else:
            widths.append(float((idx.max() - idx.min()).item()) / 2)
    return float(np.mean(widths))


def gp_cube(b, h, w, ells, gen, block=8):
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
                x[bi, :, i * block:(i + 1) * block, j * block:(j + 1) * block] = (
                    0.5 + 0.15 * s).reshape(L, block, block)
    return x.clamp(0, 1), ell_map


# ------------------------------------------------- training-free references (Part A)
def baseline_mean(inp, drop):
    """Predict the known data mean (0.5) for every dropped band."""
    return torch.where(drop, torch.full_like(inp, 0.5), inp)


def baseline_linear(inp, drop):
    """Linear interpolation along the band axis from the kept (noisy) bands."""
    out = inp.clone()
    for bi in range(inp.shape[0]):
        d = drop[bi, :, 0, 0].numpy()
        kept = np.flatnonzero(~d)
        for j in np.flatnonzero(d):
            if len(kept) == 0:
                out[bi, j] = 0.5
            else:
                out[bi, j] = torch.from_numpy(
                    np.stack([np.interp(j, kept, inp[bi, kept, y, x].numpy())
                              for y in range(inp.shape[2]) for x in range(inp.shape[3])])
                ).reshape(inp.shape[2], inp.shape[3]).float()
    return out


def baseline_gp(inp, drop, ell_pix):
    """Posterior mean under the true GP (known ell, mean 0.5, std 0.15, noise args.noise):
    the best any model can do on this task (ignoring the [0, 1] clamp)."""
    out = inp.clone()
    idx = np.arange(L)
    for ell in np.unique(ell_pix.numpy()):
        K = 0.15 ** 2 * (np.exp(-(idx[:, None] - idx[None]) ** 2 / (2 * ell ** 2)) + 1e-4 * np.eye(L))
        for bi in range(inp.shape[0]):
            d = drop[bi, :, 0, 0].numpy()
            k = ~d
            if k.all() or d.all():
                continue
            gain = K[np.ix_(d, k)] @ np.linalg.inv(K[np.ix_(k, k)] + args.noise ** 2 * np.eye(k.sum()))
            sel = (ell_pix[bi, 0] == ell)                                   # (H, W) pixels with this ell
            y = inp[bi][:, sel].numpy()                                     # (L, n)
            mu = gain @ (y[k] - 0.5) + 0.5
            tmp = out[bi][:, sel].numpy()
            tmp[d] = mu
            out[bi][:, sel] = torch.from_numpy(tmp)
    return out


def per_ell_psnr(pred, gt, drop, ell_pix):
    res = {}
    for ell in np.unique(ell_pix.numpy()):
        res[str(float(ell))] = psnr(pred.clamp(0, 1), gt, drop & (ell_pix == ell).expand_as(drop))
    res['all'] = psnr(pred.clamp(0, 1), gt, drop)
    return res


def make_sampler(pool, gen):
    return lambda b: pool[torch.randint(0, pool.shape[0], (b,), generator=gen)]


def fit_and_score(model, sampler, gt, inp, drop, tag):
    assert n_trainable_sigma(model) == 0, 'sigma must not be a trainable parameter'
    model = train(model, sampler, args.iters, tag)
    metrics, out = evaluate(model, gt, inp, drop)
    J = spectral_jacobian(model, gt[:1, :, :16, :16])
    metrics['influence_halfwidth_bands'] = band_halfwidth(J)
    metrics['params'] = n_params(model)
    print('  %-16s dropped %.2f dB | kept %.2f | half-width %.1f bands | params %d' %
          (tag, metrics['psnr_dropped'], metrics['psnr_kept'],
           metrics['influence_halfwidth_bands'], metrics['params']), flush=True)
    return metrics, out, J, model


# ----------------------------------------------------------------------------- Part A
def part_a():
    print('\n=== Part A: frozen σ + Δ=1 band on synthetic GP spectra ===')
    ells = [1.0, 2.0, 4.0, 8.0]
    gen = torch.Generator().manual_seed(args.seed)
    pool = gp_cube(512, args.crop, args.crop, ells, gen)[0]
    sampler = make_sampler(pool, gen)

    gen_test = torch.Generator().manual_seed(args.seed + 7)
    gt, _ = gp_cube(8, args.crop, args.crop, ells, gen_test)
    inp, drop = corrupt(gt, gen_test)
    print('  input PSNR: all %.2f dB, dropped bands %.2f dB' % (psnr(inp, gt), psnr(inp, gt, drop)))

    sweep, models = {}, {}
    sigmas = [1.0, 2.0, 4.0, 8.0]
    for s in sigmas:
        torch.manual_seed(args.seed)
        m = prior(s)
        print('  prior σ=%.0f  trainable-σ params = %d  (should be 0)' % (s, n_trainable_sigma(m)))
        metrics, _, J, models[str(s)] = fit_and_score(m, sampler, gt, inp, drop, 'prior_s%.0f' % s)
        sweep[str(s)] = dict(metrics=metrics, jacobian=J.tolist())

    for name, ctor in [('specMLP', lambda: Residual(SpectralMLP(L))),
                       ('spatialCNN', lambda: Residual(SpatialCNN()))]:
        torch.manual_seed(args.seed)
        metrics, _, J, models[name] = fit_and_score(ctor(), sampler, gt, inp, drop, name)
        sweep[name] = dict(metrics=metrics, jacobian=J.tolist())

    # --- per-ell evaluation against training-free references (larger test set)
    print('\n  --- per-ell evaluation (dropped-band PSNR, dB) vs. training-free references ---')
    gen_eval = torch.Generator().manual_seed(args.seed + 13)
    gt_v, ell_map_v = gp_cube(args.eval_n, args.crop, args.crop, ells, gen_eval)
    inp_v, drop_v = corrupt(gt_v, gen_eval)
    ell_pix = ell_map_v.repeat_interleave(8, 1).repeat_interleave(8, 2).unsqueeze(1)   # (b, 1, H, W)
    per_ell = {'input': per_ell_psnr(inp_v, gt_v, drop_v, ell_pix),
               'mean_0.5': per_ell_psnr(baseline_mean(inp_v, drop_v), gt_v, drop_v, ell_pix),
               'linear_interp': per_ell_psnr(baseline_linear(inp_v, drop_v), gt_v, drop_v, ell_pix),
               'gp_optimal': per_ell_psnr(baseline_gp(inp_v, drop_v, ell_pix), gt_v, drop_v, ell_pix)}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            per_ell['prior_s' + name if name[0].isdigit() else name] = per_ell_psnr(
                model(inp_v), gt_v, drop_v, ell_pix)
    cols = ['1.0', '2.0', '4.0', '8.0', 'all']
    print('  %-14s' % 'model' + ''.join('%9s' % ('ell=' + c if c != 'all' else 'all') for c in cols))
    for name, row in per_ell.items():
        print('  %-14s' % name + ''.join('%9.2f' % row[c] for c in cols), flush=True)

    print('\n  --- match / mismatch: train on a single ell ---')
    match = {}
    for ell in (2.0, 8.0):
        gen_e = torch.Generator().manual_seed(args.seed + int(ell))
        pool_e = gp_cube(256, args.crop, args.crop, [ell], gen_e)[0]
        samp_e = make_sampler(pool_e, gen_e)
        gt_e, _ = gp_cube(8, args.crop, args.crop, [ell], torch.Generator().manual_seed(99))
        inp_e, drop_e = corrupt(gt_e, torch.Generator().manual_seed(99))
        row = {}
        for s in (2.0, 8.0):
            torch.manual_seed(args.seed)
            tag = 'ell%.0f_sigma%.0f' % (ell, s)
            metrics, _, _, _ = fit_and_score(prior(s), samp_e, gt_e, inp_e, drop_e, tag)
            row[str(s)] = metrics
        match[str(ell)] = row
        print('  ell=%.0f: matched σ PSNR %.2f  vs  mismatched σ PSNR %.2f' %
              (ell, row[str(ell)]['psnr_dropped'], row[str(8.0 if ell == 2.0 else 2.0)]['psnr_dropped']))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ['σ=1', 'σ=2', 'σ=4', 'σ=8', 'specMLP', 'spatialCNN']
    keys = ['1.0', '2.0', '4.0', '8.0', 'specMLP', 'spatialCNN']
    dropped = [sweep[k]['metrics']['psnr_dropped'] for k in keys]
    widths = [sweep[k]['metrics']['influence_halfwidth_bands'] for k in keys]
    colors = ['#4C78A8'] * 4 + ['#F58518', '#54A24B']
    ax[0].bar(labels, dropped, color=colors)
    ax[0].axhline(psnr(inp, gt, drop), color='gray', ls='--', lw=1, label='corrupted input')
    ax[0].set_ylabel('dropped-band PSNR (dB)')
    ax[0].set_title('frozen-σ prior vs. baselines')
    ax[0].legend(fontsize=8)
    ax[1].bar(labels, widths, color=colors)
    ax[1].set_ylabel('influence half-width (bands)')
    ax[1].set_title('|dy_i/dx_j| half-width')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_sigma_sweep.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(2)
    matched = [match['2.0']['2.0']['psnr_dropped'], match['8.0']['8.0']['psnr_dropped']]
    mismatched = [match['2.0']['8.0']['psnr_dropped'], match['8.0']['2.0']['psnr_dropped']]
    ax.bar(x - 0.18, matched, 0.36, label='σ matches ell', color='#4C78A8')
    ax.bar(x + 0.18, mismatched, 0.36, label='σ mismatches ell', color='#E45756')
    ax.set_xticks(x)
    ax.set_xticklabels(['data ell=2', 'data ell=8'])
    ax.set_ylabel('dropped-band PSNR (dB)')
    ax.set_title('matched vs. mismatched frozen σ')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_match.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    return dict(sweep={k: v['metrics'] for k, v in sweep.items()}, match=match,
                input_psnr_dropped=psnr(inp, gt, drop), per_ell=per_ell)


# ----------------------------------------------------------------------------- Part B
def load_real():
    d = sio.loadmat(args.data)
    cube = [v for k, v in d.items() if not k.startswith('__')][0].astype(np.float32)
    nb = cube.shape[2] // L
    cube = cube[:, :, :nb * L].reshape(cube.shape[0], cube.shape[1], L, nb).mean(-1)
    cube = cube / np.percentile(cube, 99.9)
    return torch.from_numpy(cube).permute(2, 0, 1).clamp(0, 1)


def part_b():
    print('\n=== Part B: frozen σ=4, Δ=1 on Indian Pines ===')
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

    constructors = [
        ('prior_s4', lambda: prior(4.0)),
        ('specMLP', lambda: Residual(SpectralMLP(L))),
        ('spatialCNN', lambda: Residual(SpatialCNN())),
    ]
    results, outputs, jac = {}, {}, {}
    for name, ctor in constructors:
        torch.manual_seed(args.seed)
        metrics, out, J, _ = fit_and_score(ctor(), sampler, gt, inp, drop, 'real/' + name)
        results[name], outputs[name], jac[name] = metrics, out[0], J

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, name in zip(ax, [k for k, _ in constructors]):
        a.imshow(jac[name].numpy(), cmap='magma')
        a.set_title('%s\nhalf-width %.1f bands' % (name, results[name]['influence_halfwidth_bands']))
        a.set_xlabel('input band j')
        a.set_ylabel('output band i')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_real_jacobian.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    pix = [(5, 20), (16, 70), (28, 120)]
    styles = {'prior_s4': 'b-', 'specMLP': 'm:', 'spatialCNN': 'c-.'}
    for a, (r, c) in zip(ax, pix):
        x = np.arange(L)
        a.plot(x, gt[0, :, r, c].numpy(), 'k-', lw=2, label='ground truth')
        kept = ~drop[0, :, r, c]
        a.plot(x[kept.numpy()], inp[0, :, r, c][kept].numpy(), 'o', color='gray', ms=4, label='kept')
        a.plot(x[~kept.numpy()], np.zeros((~kept).sum().item()), 'x', color='red', ms=6, label='dropped')
        for name, st in styles.items():
            a.plot(x, outputs[name][:, r, c].numpy(), st, lw=1.3,
                   label='%s (%.1f dB)' % (name, results[name]['psnr_dropped']))
        a.set_xlabel('band')
        a.set_title('pixel (%d, %d)' % (r, c))
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_real_spectra.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    return results


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    path = os.path.join(args.out_dir, 'results.json')
    all_res = {}
    if os.path.exists(path):          # keep the part that is not re-run
        with open(path) as f:
            all_res = json.load(f)
    all_res['args'] = vars(args)
    if 'a' in args.parts:
        all_res['synthetic'] = part_a()
    if 'b' in args.parts:
        all_res['real'] = part_b()
    with open(path, 'w') as f:
        json.dump(all_res, f, indent=2)
    print('\nsaved to', args.out_dir)

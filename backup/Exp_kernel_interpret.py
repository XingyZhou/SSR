"""
Interpretability of the spectral token mixer.

What we measure (not reconstruction PSNR):
  1. Bare SSM impulse: B=C=1, Δ=1, D=0.  Decay vs frozen σ.
  2. Mixer impulse (output minus skip): frozen-σ SpecMamba vs CMB+SAB vs learnable A.
  3. Off-diagonal spectral Jacobian |dy_i/dx_j| (diagonal zeroed so the skip
     cannot hide the kernel).  Half-width and mean |lag| vs σ.

Usage:
    python Exp_kernel_interpret.py --out_dir ./exp_out/kernel
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from Model import SRB
from SpectralMamba import SpectralMamba, selective_scan_ref

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='./exp_out/kernel')
parser.add_argument('--bands', default=28, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--train_steps', default=200, type=int)
parser.add_argument('--threads', default=4, type=int)
args = parser.parse_args()

torch.set_num_threads(args.threads)
os.makedirs(args.out_dir, exist_ok=True)
L = args.bands
SIGMAS = (1.0, 2.0, 4.0, 8.0)


def sm_kwargs(sigma=4.0, learn_A=False):
    kw = dict(patch=4, d_state=8, width_mode='fixed', sigma_init=float(sigma),
              selective_dt=False, dt_fixed=1.0, A_mode='harmonic', fixed_A=True)
    if learn_A:
        kw.update(fixed_A=False, selective_dt=True)
    return kw


def make_mixer(kind, sigma=4.0):
    if kind == 'cmb':
        return SRB(L, spectral_mamba=False)
    if kind == 'learnA':
        return SRB(L, spectral_mamba=True, sm_kwargs=sm_kwargs(learn_A=True))
    if kind == 'frozen':
        return SRB(L, spectral_mamba=True, sm_kwargs=sm_kwargs(sigma=sigma))
    raise ValueError(kind)


def theoretical_bank(sigma, n_state=8):
    """Impulse of sum_n exp(-n t / σ), n=1..N, Δ=1."""
    t = np.arange(L)
    y = np.zeros(L)
    for n in range(1, n_state + 1):
        y += np.exp(-n * t / sigma)
    return t, y


def bare_ssm_impulse(sigma, n_state=8):
    """Scan with B=C=1, Δ=1, no D skip — the kernel implied by A/σ."""
    D, N = 4, n_state
    u = torch.zeros(1, D, L)
    u[..., 0] = 1.0
    A = -torch.arange(1, N + 1).float()[None].repeat(D, 1)
    B = torch.ones(1, N, L)
    C = torch.ones(1, N, L)
    dt = torch.ones(1, D, L)
    y = selective_scan_ref(u, dt, A, B, C, torch.full((1, D), sigma))
    return y[0, 0].detach().numpy()


def efold(y):
    """First lag where |y| drops below |y[0]|/e.  None if it never does."""
    y = np.asarray(y, dtype=np.float64)
    thr = abs(y[0]) / math.e
    hit = np.where(np.abs(y) <= thr)[0]
    return int(hit[0]) if len(hit) else None


def mean_lag_from_ir(y):
    """Energy-weighted |lag| of an impulse response (excluding t=0)."""
    y = np.abs(np.asarray(y, dtype=np.float64))
    if y[1:].sum() < 1e-12:
        return 0.0
    t = np.arange(len(y))
    w = y.copy()
    w[0] = 0.0
    return float((t * w).sum() / (w.sum() + 1e-12))


@torch.no_grad()
def mixer_impulse(model, src_band=0):
    """Spatial-mean output spectrum for a single-band impulse. Returns (raw, residual)."""
    model.eval()
    x = torch.zeros(1, L, 16, 16)
    x[:, src_band] = 1.0
    y = model(x)
    raw = y.mean(dim=(0, 2, 3)).cpu().numpy()
    residual = (y - x).mean(dim=(0, 2, 3)).cpu().numpy()
    return raw, residual


def spectral_jacobian(model, spatial=8):
    """J[i,j] = mean |dy_i / dx_j| over batch/space.  (L, L)."""
    model.eval()
    x = torch.rand(1, L, spatial, spatial, requires_grad=True)
    y = model(x)
    J = torch.zeros(L, L)
    for i in range(L):
        g, = torch.autograd.grad(y[:, i].sum(), x, retain_graph=i < L - 1)
        J[i] = g.abs().mean(dim=(0, 2, 3))
    return J.detach()


def offdiag_width(J):
    """Half-span at 10% of off-diagonal peak, and energy-weighted mean |i-j|."""
    J = J.clone()
    J.fill_diagonal_(0)
    half, com = [], []
    idx = torch.arange(L)
    for i in range(L):
        row = J[i]
        m = float(row.max())
        if m < 1e-12:
            half.append(0.0)
            com.append(0.0)
            continue
        keep = torch.nonzero(row > 0.1 * m).flatten()
        half.append(float((keep.max() - keep.min()).item()) / 2 if len(keep) else 0.0)
        w = row / (row.sum() + 1e-12)
        com.append(float((w * (idx - i).abs()).sum()))
    return dict(halfwidth=float(np.mean(half)), mean_lag=float(np.mean(com)))


def gp_batch(b=8, h=16, w=16, ell=4.0):
    idx = torch.arange(L, dtype=torch.float32)
    K = torch.exp(-(idx[:, None] - idx[None, :]) ** 2 / (2 * ell ** 2)) + 1e-4 * torch.eye(L)
    chol = torch.linalg.cholesky(K)
    z = torch.randn(L, b * h * w)
    s = 0.5 + 0.15 * (chol @ z)
    return s.T.reshape(b, L, h, w).clamp(0, 1)


def train_mixer(model, steps):
    """Band-dropout recovery so the mixer has to use neighbouring bands."""
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    g = torch.Generator().manual_seed(args.seed + 1)
    for _ in range(steps):
        gt = gp_batch(8, 16, 16, ell=4.0)
        drop = torch.rand(8, L, 1, 1, generator=g) < 0.3
        inp = gt * (~drop).float()
        loss = F.mse_loss(model(inp), gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def dump_heatmap(path, J, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(J.numpy(), cmap='magma')
    ax.set_title(title)
    ax.set_xlabel('input band j')
    ax.set_ylabel('output band i')
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = {'bare_ssm': {}, 'init': {}, 'trained': {}}

    # ----- 1. bare SSM kernel (no B/C learning, no skip) -----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    t = np.arange(L)
    for s in SIGMAS:
        y = bare_ssm_impulse(s)
        _, y_th = theoretical_bank(s)
        y_th = y_th * (y[0] / (y_th[0] + 1e-12))
        ax.plot(t, y, lw=2, label='scan σ=%g  e-fold=%s  lag=%.2f' % (s, efold(y), mean_lag_from_ir(y)))
        ax.plot(t, y_th, ls='--', alpha=0.5, color=ax.lines[-1].get_color())
        results['bare_ssm'][str(s)] = dict(ir=y.tolist(), efold=efold(y), mean_lag=mean_lag_from_ir(y))
    ax.set_xlabel('band lag')
    ax.set_ylabel('impulse response')
    ax.set_title('Bare SSM (B=C=1, Δ=1): kernel widens with σ\n dashed = Σ_n exp(-n t / σ)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_bare_ssm_ir.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ----- mixers: init + after short band-dropout training -----
    specs = [('cmb', None), ('learnA', None)] + [('frozen', s) for s in SIGMAS]
    models_init, models_tr = {}, {}
    for kind, s in specs:
        key = kind if s is None else 'frozen_s%g' % s
        torch.manual_seed(args.seed)
        models_init[key] = make_mixer(kind, sigma=s or 4.0)
        torch.manual_seed(args.seed)
        models_tr[key] = train_mixer(make_mixer(kind, sigma=s or 4.0), args.train_steps)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, store, models, slug, tag in (
            (axes[0], results['init'], models_init, 'init', 'at init'),
            (axes[1], results['trained'], models_tr, 'trained',
             'after %d dropout steps' % args.train_steps)):
        for key, m in models.items():
            _, resid = mixer_impulse(m, src_band=0)
            J = spectral_jacobian(m)
            w = offdiag_width(J)
            store[key] = dict(ir_residual=resid.tolist(), jacobian_offdiag=w,
                              ir_mean_lag=mean_lag_from_ir(resid))
            dump_heatmap(os.path.join(args.out_dir, 'jac_%s_%s.png' % (slug, key)),
                         J, '%s  %s\noff-diag lag=%.2f  half-width=%.2f' %
                         (key, tag, w['mean_lag'], w['halfwidth']))
            ax.plot(np.arange(L), resid, lw=1.6, label='%s  lag=%.2f' % (key, mean_lag_from_ir(resid)))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('output band (impulse at band 0)')
        ax.set_title('mixer residual  (%s)' % tag)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('mean (y − x)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_mixer_impulse.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ----- width vs σ (frozen only), init and trained -----
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    xs = list(SIGMAS)
    for label, store, marker in (('init mean |lag|', results['init'], 'o'),
                                 ('trained mean |lag|', results['trained'], 's')):
        ys = [store['frozen_s%g' % s]['jacobian_offdiag']['mean_lag'] for s in xs]
        ax.plot(xs, ys, marker=marker, lw=2, label=label)
    ax.plot(xs, [results['bare_ssm'][str(s)]['mean_lag'] for s in xs],
            'k--', lw=1.5, label='bare SSM mean |lag|')
    ax.set_xlabel('frozen σ (bands)')
    ax.set_ylabel('spectral mixing width (bands)')
    ax.set_title('Does mixer width track σ?  (off-diagonal Jacobian)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig_width_vs_sigma.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # compact table for the json (drop long IR lists from the printed summary)
    summary = {
        'bare_ssm_efold': {s: results['bare_ssm'][s]['efold'] for s in results['bare_ssm']},
        'bare_ssm_mean_lag': {s: results['bare_ssm'][s]['mean_lag'] for s in results['bare_ssm']},
        'init_offdiag': {k: v['jacobian_offdiag'] for k, v in results['init'].items()},
        'trained_offdiag': {k: v['jacobian_offdiag'] for k, v in results['trained'].items()},
        'init_ir_lag': {k: v['ir_mean_lag'] for k, v in results['init'].items()},
        'trained_ir_lag': {k: v['ir_mean_lag'] for k, v in results['trained'].items()},
    }
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'full': results}, f)
    print(json.dumps(summary, indent=2))
    print('saved to', args.out_dir)


if __name__ == '__main__':
    main()

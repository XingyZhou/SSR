"""Frequency-band error analysis of reconstructions on the test set.

For each model, the reconstruction error (pred - label) is split into
  1. Haar wavelet sub-bands, 2 levels: LL2, (HL2, LH2, HH2), (HL1, LH1, HH1)
  2. FFT radial rings: low / mid / high spatial frequency (equal-width rings)
and the share of error energy and the PSNR of each band are reported, so one
can see whether a model loses on low frequencies (structure) or high (detail).

    python analyze_freq.py --models SSR-S:Model.Net:./Checkpoint_pretrain/model_S.pkl \
                                    V5:model_v5.HyMCNetL3_V5:./Checkpoint_v5/model_300.pkl
"""
import argparse
import importlib
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud

from Utils import *
from Dataset import dataset
from Mamba_Model import haar_dwt2d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', required=True, help='name:module.Class:ckpt')
parser.add_argument('--out_dir', default='./Result_freq')
parser.add_argument('--data_path', default='./Test_data/')
parser.add_argument('--mask_path', default='./mask_256_28.mat')
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--stage", default=3, type=int)
parser.add_argument("--testset_num", default=10, type=int)
parser.add_argument("--trainset_num", default=5000, type=int)
parser.add_argument("--isTrain", default=False, type=bool)
parser.add_argument("--bands", default=28, type=int)
parser.add_argument("--scene_num", default=10, type=int)
parser.add_argument("--len_shift", default=2, type=int)
opt = parser.parse_args()


def prepare_testdata(file_list, n):
    HSI = np.zeros((256, 256, 28, n))
    for i in range(n):
        HSI[:, :, :, i] = sio.loadmat(file_list[i])['img']
    return np.clip(HSI, 0., 1.)


def psnr_from_mse(mse):
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def haar_bands(x):
    """x: 1xCxHxW -> dict of 2-level sub-bands (orthonormal Haar, energy preserving)."""
    ll1, hl1, lh1, hh1 = haar_dwt2d(x)
    ll2, hl2, lh2, hh2 = haar_dwt2d(ll1)
    return {'LL2': ll2, 'HL2': hl2, 'LH2': lh2, 'HH2': hh2, 'HL1': hl1, 'LH1': lh1, 'HH1': hh1}


def fft_rings(x, n_rings=3):
    """Radial FFT ring energies of x (1xCxHxW); returns list of energies, low -> high."""
    F_ = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
    H, W = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device) - H // 2,
                            torch.arange(W, device=x.device) - W // 2, indexing='ij')
    r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    edges = torch.linspace(0, r.max(), n_rings + 1)
    power = F_.abs() ** 2
    return [power[..., (r >= edges[i]) & (r < edges[i + 1] + (i == n_rings - 1))].sum().item() for i in range(n_rings)]


if __name__ == "__main__":
    os.makedirs(opt.out_dir, exist_ok=True)
    file_list = loadpath(opt.data_path + 'test_list.txt')
    file_list.sort()
    HSI = prepare_testdata(file_list, opt.scene_num)
    loader = tud.DataLoader(dataset(opt, HSI), batch_size=1, shuffle=False)

    rows = []
    for spec in opt.models:
        name, cls_path, ckpt = spec.split(':')
        mod, cls = cls_path.rsplit('.', 1)
        model = getattr(importlib.import_module(mod), cls)(opt).cuda().eval()
        model.load_state_dict(torch.load(ckpt, weights_only=False)['model'])
        band_err = {}
        band_sig = {}
        ring_err = np.zeros(3)
        ring_sig = np.zeros(3)
        n_px = 0
        with torch.no_grad():
            for g, label, Phi, Phi_s in loader:
                out = model(g, input_mask=(Phi, Phi_s))[opt.stage - 1].clamp(0, 1)
                err = out - label
                n_px += err.numel()
                for k, v in haar_bands(err).items():
                    band_err[k] = band_err.get(k, 0.) + (v ** 2).sum().item()
                for k, v in haar_bands(label).items():
                    band_sig[k] = band_sig.get(k, 0.) + (v ** 2).sum().item()
                ring_err += np.array(fft_rings(err))
                ring_sig += np.array(fft_rings(label))
        total = sum(band_err.values())
        row = {'model': name, 'PSNR': psnr_from_mse(total / n_px)}
        for k in band_err:
            row['%s_err%%' % k] = 100 * band_err[k] / total           # share of error energy
            row['%s_SNR' % k] = 10 * np.log10(band_sig[k] / band_err[k])  # band signal-to-error ratio
        for i, nm in enumerate(['low', 'mid', 'high']):
            row['fft_%s_err%%' % nm] = 100 * ring_err[i] / ring_err.sum()
            row['fft_%s_SNR' % nm] = 10 * np.log10(ring_sig[i] / ring_err[i])
        rows.append(row)
        print(name, 'done', flush=True)

    df = pd.DataFrame(rows).set_index('model')
    df.to_csv(os.path.join(opt.out_dir, 'freq_analysis.csv'))
    pd.set_option('display.width', 250)
    print("\nPSNR and share of error energy per Haar band (%):")
    print(df[['PSNR'] + [c for c in df.columns if c.endswith('_err%') and not c.startswith('fft')]].round(2))
    print("\nBand SNR (dB; higher = that band reconstructed better):")
    print(df[[c for c in df.columns if c.endswith('_SNR') and not c.startswith('fft')]].round(2))
    print("\nFFT rings: error share (%) and SNR (dB):")
    print(df[[c for c in df.columns if c.startswith('fft')]].round(2))

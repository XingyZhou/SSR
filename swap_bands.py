"""Oracle band-swap: recombine the Haar sub-bands of two models' reconstructions.

For a pair (A, B) and a Haar level L, builds hybrids
    A_low + B_high : LL_L from A, all detail bands (levels 1..L) from B
    B_low + A_high : the reverse
and reports PSNR (mean over scenes, like Test_v3.py) for A, B and both hybrids.
If "A_low + B_high" beats both A and B, A's low frequencies are genuinely
better and only its high frequencies hold it back, and vice versa.

    python swap_bands.py --a V5:model_v5.HyMCNetL3_V5:./Checkpoint_v5/model_300.pkl \
                         --b SSR-S:Model.Net:./Checkpoint_pretrain/model_S.pkl --levels 1 2 3
"""
import argparse
import importlib
import os

import numpy as np
import torch
import torch.utils.data as tud

from Utils import *
from Dataset import dataset
from Mamba_Model import haar_dwt2d, haar_idwt2d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

parser = argparse.ArgumentParser()
parser.add_argument('--a', required=True, help='name:module.Class:ckpt')
parser.add_argument('--b', required=True, help='name:module.Class:ckpt')
parser.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3])
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


def decompose(x, L):
    """-> (LL_L, [details level 1..L]) where each details entry is (hl, lh, hh)."""
    details = []
    for _ in range(L):
        x, hl, lh, hh = haar_dwt2d(x)
        details.append((hl, lh, hh))
    return x, details


def reconstruct(ll, details):
    for hl, lh, hh in reversed(details):
        ll = haar_idwt2d(ll, hl, lh, hh)
    return ll


def run_model(spec, loader):
    # name:module.Class:ckpt[:key=val,key=val]  (e.g. V7S-s4:model_v7s.HyMCNetL3_V7S:ckpt.pkl:bc_mode=s4,sigma=4)
    name, cls_path, ckpt, *extra = spec.split(':')
    mod, cls = cls_path.rsplit('.', 1)
    m_opt = argparse.Namespace(**vars(opt))
    for kv in (extra[0].split(',') if extra else []):
        k, v = kv.split('=')
        setattr(m_opt, k, float(v) if k in ('sigma', 'dt_fixed') else v)
    model = getattr(importlib.import_module(mod), cls)(m_opt).cuda().eval()
    model.load_state_dict(torch.load(ckpt, weights_only=False)['model'])
    outs, labels = [], []
    with torch.no_grad():
        for g, label, Phi, Phi_s in loader:
            outs.append(model(g, input_mask=(Phi, Phi_s))[opt.stage - 1].clamp(0, 1))
            labels.append(label)
    return name, outs, labels


def psnr(outs, labels):
    return np.mean([compare_psnr(l.cpu().numpy(), o.clamp(0, 1).cpu().numpy(), data_range=1.0)
                    for o, l in zip(outs, labels)])


if __name__ == "__main__":
    file_list = loadpath(opt.data_path + 'test_list.txt')
    file_list.sort()
    loader = tud.DataLoader(dataset(opt, prepare_testdata(file_list, opt.scene_num)), batch_size=1, shuffle=False)
    na, A, labels = run_model(opt.a, loader)
    nb, B, _ = run_model(opt.b, loader)
    print("%-28s PSNR %.3f" % (na, psnr(A, labels)))
    print("%-28s PSNR %.3f" % (nb, psnr(B, labels)))
    for L in opt.levels:
        ab, ba = [], []
        for a, b in zip(A, B):
            la, da = decompose(a, L)
            lb, db = decompose(b, L)
            ab.append(reconstruct(la, db))
            ba.append(reconstruct(lb, da))
        res = max(a.shape[-1] >> L for a in A[:1])
        print("level %d (LL at %3dx%-3d): %s_low + %s_high  PSNR %.3f   |   %s_low + %s_high  PSNR %.3f" % (
            L, res, res, na, nb, psnr(ab, labels), nb, na, psnr(ba, labels)))

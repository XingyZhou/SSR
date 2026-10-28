"""Test_nb: Test_v3.py generalised to any band count (models from model_nb.py, data via Dataset_nb).

Examples:
    .venv/bin/python Test_nb.py --model HyMCNetL3_V7NB --data_path ./Chikusei_128/ --bands 128 \
        --mask_path ./mask_256_128.mat --ckpt_dir ./Checkpoint_chik_v7nb --out_dir ./Result_chik_v7nb --epochs 291-300

Outputs (under --out_dir): summary.csv, per_scene.csv, report.md, best.json, recon/<epoch>/N.mat (key 'hsi').
Test scenes are listed in <data_path>/test_list.txt and must be size x size x bands (key 'img', float in [0,1]).
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.utils.data as tud

import model_nb
from Dataset_nb import dataset_nb, load_scenes
from Test_v3 import ssim, sam
from Utils import compare_psnr, findLastCheckpoint, loadpath

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

parser = argparse.ArgumentParser(description="Band-count-agnostic CASSI test")
parser.add_argument('--model', default='HyMCNetL3_V7NB', type=str)
parser.add_argument('--data_path', default='./Chikusei_128/', type=str)
parser.add_argument('--test_list', default='test_list.txt', type=str)
parser.add_argument('--mask_path', default='./mask_256_128.mat', type=str)
parser.add_argument('--ckpt_dir', default='./Checkpoint_chik_v7nb', type=str)
parser.add_argument('--out_dir', default='./Result_chik_v7nb', type=str)
parser.add_argument('--epochs', default='last', type=str,
                    help="'last', 'best-train', a number, a range '291-300', or a list '250,280,300'")
parser.add_argument('--save_recon', default=1, type=int)
parser.add_argument("--bands", default=128, type=int)
parser.add_argument("--mult", default='1,2,4', type=str)
parser.add_argument("--win", default=8, type=int)
parser.add_argument("--len_shift", default=2, type=int)
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--stage", default=3, type=int)
parser.add_argument("--trainset_num", default=5000, type=int)
parser.add_argument("--testset_num", default=0, type=int, help='0 = all scenes in test_list')
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--isTrain", default=False, type=bool)
parser.add_argument("--bc_mode", default='selective', choices=['selective', 's4', 'const'])
parser.add_argument("--sigma", default=4.0, type=float)
parser.add_argument("--dt_fixed", default=1.0, type=float)
opt = parser.parse_args()


def parse_epochs(spec, ckpt_dir):
    last = findLastCheckpoint(ckpt_dir)
    if spec == 'last':
        return [last]
    if spec == 'best-train':
        best, best_psnr = last, -1
        for f in glob.glob(os.path.join(ckpt_dir, 'model_*.pkl')):
            ck = torch.load(f, map_location='cpu', weights_only=False)
            if ck.get('psnr', -1) > best_psnr:
                best, best_psnr = ck['epoch'], ck['psnr']
        return [best]
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    os.makedirs(opt.out_dir, exist_ok=True)

    file_list = loadpath(os.path.join(opt.data_path, opt.test_list))
    file_list.sort()
    if opt.testset_num > 0:
        file_list = file_list[:opt.testset_num]
    opt.testset_num = len(file_list)
    names = [os.path.splitext(os.path.basename(p))[0] for p in file_list]
    HSI = load_scenes(file_list, opt.bands)
    loader = tud.DataLoader(dataset_nb(opt, HSI), batch_size=1, shuffle=False, num_workers=0)

    model = getattr(model_nb, opt.model)(opt).cuda()
    epochs = parse_epochs(opt.epochs, opt.ckpt_dir)
    print("Evaluating epochs:", epochs, flush=True)

    summary, per_scene, recons = [], [], {}
    with torch.no_grad():
        for ep in epochs:
            ckpt = torch.load(os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % ep), weights_only=False)
            model.load_state_dict(ckpt['model'])
            model.eval()
            cubes, stage_psnr, times = [], [[] for _ in range(opt.stage)], []
            for step, (g, label, Phi, Phi_s) in enumerate(loader):
                torch.cuda.synchronize()
                t0 = time.time()
                outs = model(g, input_mask=(Phi, Phi_s))
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                for s in range(opt.stage):
                    o = outs[s].clamp(0., 1.)
                    stage_psnr[s].append(compare_psnr(label.cpu().numpy(), o.cpu().numpy(), data_range=1.0))
                out = outs[opt.stage - 1].clamp(0., 1.)
                per_scene.append({'epoch': ep, 'scene': names[step], 'PSNR': stage_psnr[opt.stage - 1][-1],
                                  'SSIM': ssim(out, label), 'SAM': sam(out, label)})
                cubes.append(out.squeeze(0).permute(1, 2, 0).cpu().numpy())
            rows = [r for r in per_scene if r['epoch'] == ep]
            row = {'epoch': ep,
                   'test_PSNR': np.mean([r['PSNR'] for r in rows]),
                   'test_SSIM': np.mean([r['SSIM'] for r in rows]),
                   'test_SAM': np.mean([r['SAM'] for r in rows]),
                   'train_PSNR': ckpt.get('psnr', float('nan')),
                   'ms_per_scene': 1000 * np.mean(times)}
            for s in range(opt.stage):
                row['stage%d_PSNR' % (s + 1)] = np.mean(stage_psnr[s])
            summary.append(row)
            recons[ep] = cubes
            print("epoch %3d  PSNR %.4f  SSIM %.4f  SAM %.3f  (train PSNR %.4f)" % (
                ep, row['test_PSNR'], row['test_SSIM'], row['test_SAM'], row['train_PSNR']), flush=True)

    df = pd.DataFrame(summary)
    df_scene = pd.DataFrame(per_scene)
    df.to_csv(os.path.join(opt.out_dir, 'summary.csv'), index=False)
    df_scene.to_csv(os.path.join(opt.out_dir, 'per_scene.csv'), index=False)

    best = int(df.loc[df['test_PSNR'].idxmax(), 'epoch'])
    if opt.save_recon:
        d = os.path.join(opt.out_dir, 'recon', '%03d' % best)
        os.makedirs(d, exist_ok=True)
        for i, cube in enumerate(recons[best]):
            sio.savemat(os.path.join(d, '%d.mat' % (i + 1)), {'hsi': cube})

    b = df[df['epoch'] == best].iloc[0]
    bs = df_scene[df_scene['epoch'] == best].set_index('scene')
    with open(os.path.join(opt.out_dir, 'report.md'), 'w') as f:
        f.write("# %s test report (%d bands, %s)\n\n" % (opt.model, opt.bands, opt.data_path))
        f.write("- checkpoint dir: `%s`\n- epochs evaluated: %s\n- best epoch (by test PSNR): **%d**\n\n"
                % (opt.ckpt_dir, epochs, best))
        f.write("| metric | value |\n|---|---|\n")
        f.write("| PSNR (dB) | %.4f |\n| SSIM | %.4f |\n| SAM (deg) | %.3f |\n" % (
            b['test_PSNR'], b['test_SSIM'], b['test_SAM']))
        f.write("| train PSNR (dB) | %.4f |\n| inference | %.1f ms/scene |\n\n" % (b['train_PSNR'], b['ms_per_scene']))
        f.write("Per-stage PSNR: " + ", ".join("stage%d %.3f" % (s + 1, b['stage%d_PSNR' % (s + 1)])
                                                 for s in range(opt.stage)) + "\n\n")
        f.write("## Per scene\n\n| scene | PSNR | SSIM | SAM |\n|---|---|---|---|\n")
        for n, r in bs.iterrows():
            f.write("| %s | %.3f | %.4f | %.3f |\n" % (n, r['PSNR'], r['SSIM'], r['SAM']))
        f.write("| **mean** | **%.3f** | **%.4f** | **%.3f** |\n" % (b['test_PSNR'], b['test_SSIM'], b['test_SAM']))
        if len(df) > 1:
            f.write("\n## All epochs\n\n" + df.round(4).to_markdown(index=False) + "\n")
    with open(os.path.join(opt.out_dir, 'best.json'), 'w') as f:
        json.dump({k: (float(v) if k != 'epoch' else int(v)) for k, v in b.items()}, f, indent=2)
    print("\nBest epoch %d: PSNR %.4f SSIM %.4f SAM %.3f" % (best, b['test_PSNR'], b['test_SSIM'], b['test_SAM']))
    print("Results written to", opt.out_dir)

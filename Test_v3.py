"""Test HyMCNetL3_V3 checkpoints on the 10 test scenes and aggregate the results.

Examples:
    python Test_v3.py                        # evaluate the last checkpoint
    python Test_v3.py --epochs 291-300       # evaluate a range, pick the best by test PSNR
    python Test_v3.py --epochs 250,280,300

Outputs (under --out_dir, default ./Result_v3):
    summary.csv        one row per evaluated epoch (mean PSNR / SSIM / SAM, train PSNR)
    per_scene.csv      per-scene metrics for every evaluated epoch
    report.md          readable summary of the best epoch
    recon/<epoch>/N.mat    reconstructed cubes (--save_recon), key 'hsi', shape HxWx28
"""
import argparse
import json
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as tud
from Utils import *
import importlib
from Dataset import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

parser = argparse.ArgumentParser(description="Test HyMCNetL3_V3 on the test set")
parser.add_argument('--data_path', default='./Test_data/', type=str, help='Path of data')
parser.add_argument('--mask_path', default='./mask_256_28.mat', type=str, help='Path of mask')
parser.add_argument('--ckpt_dir', default='./Checkpoint_v3', type=str, help='Checkpoint directory')
parser.add_argument('--out_dir', default='./Result_v3', type=str, help='Where to write results')
parser.add_argument('--model', default='HyMCNetL3_V3', type=str, help='Class name in Mamba_Model or model_v4')
parser.add_argument('--epochs', default='last', type=str,
                    help="'last', 'best-train', a number, a range '291-300', or a list '250,280,300'")
parser.add_argument('--save_recon', default=1, type=int, help='Save reconstructions of the best epoch (1/0)')
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--stage", default=3, type=int, help='Number of unfolding stages')
parser.add_argument("--trainset_num", default=5000, type=int)
parser.add_argument("--testset_num", default=10, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--isTrain", default=False, type=bool)
parser.add_argument("--bands", default=28, type=int)
parser.add_argument("--scene_num", default=10, type=int)
parser.add_argument("--len_shift", default=2, type=int)
# ---- HyMCNetL3_V7S only: must match the values the checkpoint was trained with (Train_v7s.py)
parser.add_argument("--bc_mode", default='selective', choices=['selective', 's4', 'const'])
parser.add_argument("--sigma", default=4.0, type=float)
parser.add_argument("--dt_fixed", default=1.0, type=float)
opt = parser.parse_args()


def prepare_testdata(file_list, file_num):
    HSI = np.zeros((256, 256, 28, file_num))
    for idx in range(file_num):
        data = sio.loadmat(file_list[idx])
        HSI[:, :, :, idx] = data['img']
    HSI[HSI < 0.] = 0.
    HSI[HSI > 1.] = 1.
    return HSI


def _gaussian_window(size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(pred, ref, data_range=1.0):
    """Mean SSIM over bands (Gaussian window 11, sigma 1.5). pred/ref: 1xCxHxW tensors."""
    c = pred.shape[1]
    w = _gaussian_window(device=pred.device).repeat(c, 1, 1, 1)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu1 = F.conv2d(pred, w, groups=c)
    mu2 = F.conv2d(ref, w, groups=c)
    s11 = F.conv2d(pred * pred, w, groups=c) - mu1 ** 2
    s22 = F.conv2d(ref * ref, w, groups=c) - mu2 ** 2
    s12 = F.conv2d(pred * ref, w, groups=c) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.mean().item()


def sam(pred, ref, eps=1e-8):
    """Mean spectral angle in degrees. pred/ref: 1xCxHxW tensors."""
    dot = (pred * ref).sum(1)
    n = pred.norm(dim=1) * ref.norm(dim=1) + eps
    return torch.rad2deg(torch.acos((dot / n).clamp(-1, 1))).mean().item()


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

    file_list = loadpath(opt.data_path + 'test_list.txt')
    file_list.sort()
    names = [os.path.splitext(os.path.basename(p))[0] for p in file_list[:opt.scene_num]]
    HSI = prepare_testdata(file_list, opt.scene_num)
    loader = tud.DataLoader(dataset(opt, HSI), batch_size=1, shuffle=False, num_workers=0)

    model_cls = None
    for mod in ('model_v7s', 'model_v7', 'model_v6', 'model_v5', 'model_baseline', 'model_v4', 'Mamba_Model'):
        model_cls = getattr(importlib.import_module(mod), opt.model, None)
        if model_cls is not None:
            break
    model = model_cls(opt).cuda()
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
                p = stage_psnr[opt.stage - 1][-1]
                per_scene.append({'epoch': ep, 'scene': names[step], 'PSNR': p,
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
        f.write("# %s test report\n\n" % opt.model)
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

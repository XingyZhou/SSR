"""Train_nb: Train_v7.py generalised to any band count / dataset (see model_nb.py, Dataset_nb.py).

Recipe (Adam 4e-4, cosine to 1e-6 over 300 epochs, RMSE multi-stage loss, 5000 samples/epoch,
batch 2, 256x256 patches) is the one used by every root Train_v*.py, so CAVE-28 numbers and
100+ band numbers are produced by the same loop.

Examples:
    # Chikusei 128 bands, spectral scan (native S6)
    .venv/bin/python Train_nb.py --model HyMCNetL3_V7NB --data_path ./Chikusei_128/ --bands 128 \
        --mask_path ./mask_256_128.mat --ckpt_dir ./Checkpoint_chik_v7nb
    # same data, spatial-scan control
    .venv/bin/python Train_nb.py --model HyMCNetL3_NB --data_path ./Chikusei_128/ --bands 128 \
        --mask_path ./mask_256_128.mat --ckpt_dir ./Checkpoint_chik_nb
    # CAVE 28 sanity check (should reproduce V7)
    .venv/bin/python Train_nb.py --model HyMCNetL3_V7NB --data_path ./CAVE_1024_28/ --bands 28 \
        --mask_path ./mask_256_28.mat --ckpt_dir ./Checkpoint_cave_v7nb
"""
import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.utils.data as tud
from torch import optim

import model_nb
from Dataset_nb import dataset_nb, load_scenes
from Utils import compare_psnr, findLastCheckpoint, loadpath

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

parser = argparse.ArgumentParser(description="Band-count-agnostic CASSI training")
parser.add_argument('--model', default='HyMCNetL3_V7NB', type=str, help='class name in model_nb.py')
parser.add_argument('--data_path', default='./Chikusei_128/', type=str, help='dir holding train_list.txt')
parser.add_argument('--train_list', default='train_list.txt', type=str)
parser.add_argument('--mask_path', default='./mask_256_128.mat', type=str)
parser.add_argument('--ckpt_dir', default='./Checkpoint_chik_v7nb', type=str)
parser.add_argument("--bands", default=128, type=int, help='number of spectral bands')
parser.add_argument("--mult", default='1,2,4', type=str, help='U-Net width multipliers (level1,level2,bottleneck)')
parser.add_argument("--win", default=8, type=int, help='spectral-scan window side (d_model = win*win)')
parser.add_argument("--len_shift", default=2, type=int, help='CASSI dispersion step in pixels per band')
parser.add_argument("--size", default=256, type=int)
parser.add_argument("--stage", default=3, type=int)
parser.add_argument("--trainset_num", default=5000, type=int)
parser.add_argument("--testset_num", default=5, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--isTrain", default=True, type=bool)
parser.add_argument("--lr", default=0.0004, type=float)
parser.add_argument("--max_epochs", default=300, type=int, help='stop after this epoch (schedule stays 300)')
parser.add_argument("--sched_epochs", default=300, type=int, help='cosine schedule length')
# FixedSigmaMamba arm only
parser.add_argument("--bc_mode", default='selective', choices=['selective', 's4', 'const'])
parser.add_argument("--sigma", default=4.0, type=float)
parser.add_argument("--dt_fixed", default=1.0, type=float)
opt = parser.parse_args()


def loss_f(loss_func, pred, lbl):
    return torch.sqrt(loss_func(pred, lbl))


if __name__ == "__main__":
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    print(opt)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    model = getattr(model_nb, opt.model)(opt).cuda()
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('%s: %.3f M trainable parameters' % (opt.model, n_param / 1e6))

    print('time = %s' % (datetime.datetime.now()))
    file_list = loadpath(os.path.join(opt.data_path, opt.train_list))
    file_list.sort()
    HSI = load_scenes(file_list, opt.bands)
    print('loaded %d training scenes, e.g. %s' % (len(HSI), HSI[0].shape))

    Dataset = dataset_nb(opt, HSI)
    loader_train = tud.DataLoader(Dataset, batch_size=opt.batch_size, shuffle=True)
    print('time = %s' % (datetime.datetime.now()))

    mse = torch.nn.MSELoss().cuda()
    start_epoch = findLastCheckpoint(save_dir=opt.ckpt_dir)
    resume_opt = None
    if start_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % start_epoch)
        checkpoint = torch.load(os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % start_epoch), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        resume_opt = checkpoint.get('optimizer')
        start_epoch = 1 + checkpoint['epoch']
    else:
        start_epoch = 1
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr,
                           betas=(0.9, 0.999), eps=1e-8)
    if start_epoch > 1 and resume_opt is not None:
        optimizer.load_state_dict(resume_opt)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.sched_epochs, eta_min=1e-6,
                                                           last_epoch=start_epoch - 2)
    model.train()
    print('time = %s' % (datetime.datetime.now()))
    for epoch in range(opt.sched_epochs):
        if start_epoch + epoch > opt.max_epochs:
            break
        epoch_loss = 0
        psnr_total = 0
        start_time = time.time()
        for step, (g, label, Phi_batch, Phi_s_batch) in enumerate(loader_train):
            out = model(g=g, input_mask=(Phi_batch, Phi_s_batch))

            psnr = compare_psnr(label.detach().cpu().numpy(), out[opt.stage - 1].detach().cpu().numpy(),
                                data_range=1.0)
            psnr_total = psnr_total + psnr

            # same weighting as Train_v*.py (with stage=3, out[stage-4] is out[-1], the last stage again)
            loss = loss_f(mse, out[opt.stage - 1], label) + 0.7 * loss_f(mse, out[opt.stage - 2], label) + \
                0.5 * loss_f(mse, out[opt.stage - 3], label) + 0.3 * loss_f(mse, out[opt.stage - 4], label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print('%4d %4d / %4d loss = %.16f time = %s' % (
                    start_epoch + epoch, step, len(Dataset) // opt.batch_size,
                    epoch_loss / ((step + 1) * opt.batch_size),
                    datetime.datetime.now()), flush=True)

        elapsed_time = time.time() - start_time
        scheduler.step()
        print('epoch = %4d , loss = %.16f , Avg PSNR = %.4f ,time = %4.2f s' % (
            start_epoch + epoch, epoch_loss / len(Dataset), psnr_total / (step + 1), elapsed_time), flush=True)
        state = {'model': model.state_dict(), 'epoch': start_epoch + epoch,
                 'optimizer': optimizer.state_dict(), 'opt': vars(opt),
                 'loss': epoch_loss / len(Dataset), 'psnr': psnr_total / (step + 1)}
        torch.save(state, os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % (start_epoch + epoch)))

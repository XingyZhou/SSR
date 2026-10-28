"""
Train.py 的可切换副本:模型换成 Model_SpecMamba.Net(Model.py 的 Net + SRB 内光谱维 FixedSigmaMamba)。
原 Train.py / Model.py 不动。在仓库根目录运行:

    python SpectralMambaNative/Train_SpecMamba.py --bc_mode selective   # 半选择(默认)
    python SpectralMambaNative/Train_SpecMamba.py --bc_mode s4          # 物理 S4
    python SpectralMambaNative/Train_SpecMamba.py --bc_mode const       # 固定核下界
    python SpectralMambaNative/Train_SpecMamba.py --bc_mode none        # 不加 SpecMamba 的原版 Net 基线

权重默认写到 SpectralMambaNative/Checkpoint_<bc_mode>_sigma<σ>/,支持断点续训。
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import torch.utils.data as tud
from torch import optim
import time
import datetime
import argparse
import torch
import torch.nn as nn
from Utils import *
from Model_SpecMamba import Net
from Dataset import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

## Model Config
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging (SpectralMambaNative)")
parser.add_argument('--data_path', default=os.path.join(_ROOT, 'CAVE_1024_28') + '/', type=str, help='Path of data')
parser.add_argument('--mask_path', default=os.path.join(_ROOT, 'mask_256_28.mat'), type=str, help='Path of mask')
parser.add_argument("--size", default=256, type=int, help='The training image size')
parser.add_argument("--stage", default=3, type=int, help='Model scale')
parser.add_argument("--trainset_num", default=5000, type=int, help='The number of training samples of each epoch')
parser.add_argument("--testset_num", default=5, type=int, help='Total number of testset')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=2, type=int, help='Batch_size')
parser.add_argument("--isTrain", default=True, type=bool, help='Train or test')
parser.add_argument("--bands", default=28, type=int, help='The number of channels of Datasets')
parser.add_argument("--scene_num", default=205, type=int, help='The number of scenes of Datasets')
parser.add_argument("--lr", default=0.0004, type=float, help='learning rate')
parser.add_argument("--len_shift", default=2, type=int, help=' shift length among bands')
parser.add_argument("--epochs", default=300, type=int, help='Total epochs')
# ---- SpectralMambaNative
parser.add_argument("--bc_mode", default='selective', choices=['selective', 's4', 'const', 'none'],
                    help="B/C 选择性: selective | s4 | const; none = 原版 Net 基线")
parser.add_argument("--sigma", default=4.0, type=float, help='固定 A 的光谱宽度先验 (波段数)')
parser.add_argument("--dt_fixed", default=1.0, type=float, help='固定 dt (波段间距)')
parser.add_argument("--patch", default=4, type=int, help='空间块边长 p, d_model = p*p')
parser.add_argument("--d_state", default=8, type=int, help='SSM 状态数 N')
parser.add_argument("--ckpt_dir", default=None, type=str, help='默认 SpectralMambaNative/Checkpoint_<bc_mode>_sigma<σ>')
opt = parser.parse_args()
if opt.ckpt_dir is None:
    tag = opt.bc_mode if opt.bc_mode == 'none' else f"{opt.bc_mode}_sigma{opt.sigma:g}"
    opt.ckpt_dir = os.path.join(_HERE, f"Checkpoint_{tag}")
os.makedirs(opt.ckpt_dir, exist_ok=True)


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

    model = Net(opt)
    model.cuda()
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params = %.4f M' % (n_param / 1e6))

    print('time = %s' % (datetime.datetime.now()))
    ## Load training data
    key = 'train_list.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    file_list.sort()
    HSI = prepare_data(file_list, opt.scene_num)

    Dataset = dataset(opt, HSI)
    loader_train = tud.DataLoader(Dataset, batch_size=opt.batch_size, shuffle=True)
    print('time = %s' % (datetime.datetime.now()))

    mse = torch.nn.MSELoss().cuda()
    ## Load trained model
    start_epoch = findLastCheckpoint(save_dir=opt.ckpt_dir)
    if start_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % start_epoch)
        checkpoint = torch.load(os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % start_epoch), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        start_epoch = 1 + checkpoint['epoch']
    else:
        start_epoch = 1
    optimizer = optim.Adam([{'params': [p for p in model.parameters() if p.requires_grad], 'initial_lr': opt.lr}],
                           lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-6, last_epoch=start_epoch - 2)
    model.train()
    print('time = %s' % (datetime.datetime.now()))
    ## pipline of training
    for epoch in range(opt.epochs - (start_epoch - 1)):
        epoch_loss = 0
        psnr_total = 0
        start_time = time.time()
        for step, (g, label, Phi_batch, Phi_s_batch) in enumerate(loader_train):

            out = model(g=g, input_mask=(Phi_batch, Phi_s_batch))

            psnr = compare_psnr(label.detach().cpu().numpy(), out[opt.stage-1].detach().cpu().numpy(), data_range=1.0)
            psnr_total = psnr_total + psnr

            # 与 Train.py 保持相同的多阶段损失权重
            loss = loss_f(mse, out[opt.stage-1], label) + 0.7 * loss_f(mse, out[opt.stage-2], label) + \
                   0.5 * loss_f(mse, out[opt.stage-3], label) + 0.3 * loss_f(mse, out[opt.stage-4], label)
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
                 'loss': epoch_loss / len(Dataset), 'psnr': psnr_total / (step + 1), 'opt': vars(opt)}
        torch.save(state, os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % (start_epoch + epoch)))

# [备份注释] 文件名: Train.py
# [备份注释] 作用: 训练脚本 (CAVE 数据集)
# [备份注释] 备份来源: 提交 5da19fd 时的原文件, 内容未改动

import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
from Utils import *
from Model import Net
from Dataset import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Model Config
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
parser.add_argument('--data_path', default='./CAVE_1024_28/', type=str, help='Path of data')
parser.add_argument('--mask_path', default='./mask_256_28.mat', type=str, help='Path of mask')
parser.add_argument("--size", default=256, type=int, help='The training image size')
parser.add_argument("--stage", default=9, type=int, help='Unfolding stage number')
parser.add_argument("--trainset_num", default=5000, type=int, help='The number of training samples of each epoch')
parser.add_argument("--testset_num", default=5, type=int, help='Total number of testset')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=2, type=int, help='Batch_size')
parser.add_argument("--isTrain", default=True, type=bool, help='Train or test')
parser.add_argument("--bands", default=28, type=int, help='The number of channels of Datasets')
parser.add_argument("--scene_num", default=205, type=int, help='The number of scenes of Datasets') ## 205 
parser.add_argument("--lr", default=0.0004, type=float, help='learning rate')
parser.add_argument("--epochs", default=300, type=int, help='Number of training epochs')
parser.add_argument("--ckpt_dir", default="./Checkpoint", type=str, help='Checkpoint directory')
parser.add_argument("--ip_path", default="/tmp/hsi/ip.mat", type=str, help='Indian Pines fallback if CAVE files are missing')
parser.add_argument("--len_shift", default=2, type=int, help=' shift length among bands')
parser.add_argument("--spectral_mamba", action='store_true', help='Replace CMB+SAB with SpecMamba as the spectral token mixer (dim==bands only)')
parser.add_argument("--sm_width_mode", default='fixed', choices=['fixed', 'estimate', 'param'], help='fixed: frozen sigma hyperparameter; estimate/param: learned width')
parser.add_argument("--sm_sigma", default=4.0, type=float, help='Spectral width in bands (frozen when sm_width_mode=fixed)')
parser.add_argument("--sm_patch", default=4, type=int, help='Spatial patch size of one spectral sequence token')
parser.add_argument("--sm_d_state", default=8, type=int, help='SSM state size N (number of exponential kernels)')
parser.add_argument("--sm_fixed_dt", action='store_true', default=True, help='Fix the SSM step to sm_dt bands (default on)')
parser.add_argument("--sm_dt", default=1.0, type=float, help='Physical SSM step in bands when dt is fixed')
parser.add_argument("--sm_A_mode", default='harmonic', choices=['harmonic', 'uniform'], help='harmonic: A=-(n+1)/sigma; uniform: A=-1/sigma')
opt = parser.parse_args()


def loss_f(loss_func, pred, lbl):
    return torch.sqrt(loss_func(pred, lbl))


def stage_loss(mse, out, label, n_stage):
    weights = (1.0, 0.7, 0.5, 0.3)
    loss = 0
    for i, w in enumerate(weights[:n_stage]):
        loss = loss + w * loss_f(mse, out[n_stage - 1 - i], label)
    return loss


if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    device = get_device()
    if device.type == 'cuda':
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    print(opt)
    print('device =', device)

    os.makedirs(opt.ckpt_dir, exist_ok=True)
    model = Net(opt).to(device)

    print('time = %s' % (datetime.datetime.now()))
    ## Load training data
    key = 'train_list.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path) if os.path.isfile(file_path) else []
    file_list.sort()
    if cave_files_exist(file_list):
        HSI = prepare_data(file_list, opt.scene_num)
        print('loaded CAVE: %s scenes %s' % (opt.scene_num, HSI.shape))
    else:
        HSI = prepare_indian_pines(opt.ip_path, n_bands=opt.bands)
        opt.scene_num = HSI.shape[-1]
        print('CAVE missing; loaded Indian Pines fallback %s  scenes=%d' % (HSI.shape, opt.scene_num))

    Dataset = dataset(opt, HSI)
    loader_train = tud.DataLoader(Dataset, batch_size=opt.batch_size, shuffle=True)
    print('time = %s' % (datetime.datetime.now()))

    mse = torch.nn.MSELoss().to(device)
    ## Load trained model
    start_epoch = findLastCheckpoint(save_dir=opt.ckpt_dir)
    if start_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % start_epoch)
        checkpoint = torch.load(os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % start_epoch),
                                map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = 1 + checkpoint['epoch']
    else:
        start_epoch = 1
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-6,
                                                           last_epoch=start_epoch - 2)
    model.train()
    print('time = %s' % (datetime.datetime.now()))
    log_path = os.path.join(opt.ckpt_dir, 'train_log.txt')
    ## pipline of training
    for epoch in range(opt.epochs):
        epoch_loss = 0
        psnr_total = 0
        sam_total = 0
        start_time = time.time()
        for step, (g, label, Phi_batch, Phi_s_batch) in enumerate(loader_train):

            out = model(g=g, input_mask=(Phi_batch, Phi_s_batch))
            pred = out[opt.stage - 1]

            psnr = compare_psnr(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), data_range=1.0)
            sam = compare_sam(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
            psnr_total = psnr_total + psnr
            sam_total = sam_total + sam

            loss = stage_loss(mse, out, label, opt.stage)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                msg = '%4d %4d / %4d loss = %.16f psnr = %.3f sam = %.3f time = %s' % (
                    start_epoch + epoch, step, len(Dataset) // opt.batch_size,
                    epoch_loss / ((step + 1) * opt.batch_size), psnr, sam,
                    datetime.datetime.now())
                print(msg)
                with open(log_path, 'a') as f:
                    f.write(msg + '\n')

        elapsed_time = time.time() - start_time
        scheduler.step()
        avg_psnr = psnr_total / (step + 1)
        avg_sam = sam_total / (step + 1)
        msg = 'epoch = %4d , loss = %.16f , Avg PSNR = %.4f , Avg SAM = %.4f deg ,time = %4.2f s' % (
            start_epoch + epoch, epoch_loss / len(Dataset), avg_psnr, avg_sam, elapsed_time)
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
        state = {'model': model.state_dict(), 'epoch': start_epoch + epoch,
                 'loss': epoch_loss / len(Dataset), 'psnr': avg_psnr, 'sam': avg_sam}
        torch.save(state, os.path.join(opt.ckpt_dir, 'model_%03d.pkl' % (start_epoch + epoch)))

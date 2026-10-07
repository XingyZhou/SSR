from fvcore.nn import FlopCountAnalysis
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
from thop import profile
from Utils import *
from Model import Net
from Dataset import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Model Config
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
parser.add_argument("--size", default=256, type=int, help='The training image size')
parser.add_argument("--stage", default=9, type=int, help='Net stage number')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=5, type=int, help='Batch_size')
parser.add_argument("--bands", default=28, type=int, help='The number of channels of Datasets')
parser.add_argument("--lr", default=0.0001, type=float, help='learning rate')
parser.add_argument("--len_shift", default=2, type=int, help=' shift length among bands')
parser.add_argument("--spectral_mamba", action='store_true', help='Add the spectral Mamba branch (fixed A from spectral width) to SRB')
parser.add_argument("--sm_width_mode", default='fixed', choices=['fixed', 'estimate', 'param'], help='fixed: frozen sigma hyperparameter; estimate/param: learned width')
parser.add_argument("--sm_sigma", default=4.0, type=float, help='Spectral width in bands (frozen when sm_width_mode=fixed)')
parser.add_argument("--sm_patch", default=4, type=int, help='Spatial patch size of one spectral sequence token')
parser.add_argument("--sm_d_state", default=8, type=int, help='SSM state size N (number of exponential kernels)')
parser.add_argument("--sm_fixed_dt", action='store_true', default=True, help='Fix the SSM step to sm_dt bands (default on)')
parser.add_argument("--sm_dt", default=1.0, type=float, help='Physical SSM step in bands when dt is fixed')
parser.add_argument("--sm_A_mode", default='harmonic', choices=['harmonic', 'uniform'], help='harmonic: A=-(n+1)/sigma; uniform: A=-1/sigma')
opt = parser.parse_args()

if __name__ == "__main__":

    ##Calculate Parameters and FLOPs
    model = Net(opt)
    model.cuda()
    model.eval()

    input1 = torch.randn([1, 1, 256, 310]).cuda()
    input2 = torch.randn(1, 28, 256, 256).cuda()
    input3 = torch.randn(1, 256, 310).cuda()
    with torch.no_grad():
        # flops, params = profile(model, inputs=(input1, (input2,input3)))
        # print("FLOPs=", str(flops / (1024*1024*1024)) + '{}'.format("G"))
        # print("params=", str(params / (1024*1024)) + '{}'.format("M"))

        flops = FlopCountAnalysis(model, inputs=(input1, (input2, input3)))
        n_param = sum([p.nelement() for p in model.parameters()])
        print("FLOPs=", str(flops.total() / (1024 * 1024 * 1024)) + '{}'.format("G"))
        print("params=", str(n_param / (1024 * 1024)) + '{}'.format("M"))


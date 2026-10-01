import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random
import pandas as pd


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def shift_back(x, len_shift=2, bands=28):
    _, _, row, _ = x.shape
    for i in range(bands):
        x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
    return x[:, :, :, :row]

def shift_4(f, len_shift=0):
    [bs, nC, row, col] = f.shape
    shift_f = torch.zeros(bs, nC, row, col + (nC - 1) * len_shift, device=f.device, dtype=torch.float32)
    for c in range(nC):
        shift_f[:, c, :, c * len_shift:c * len_shift + col] = f[:, c, :, :]
    return shift_f


def shift_3(f, len_shift=0):
    [nC, row, col] = f.shape
    shift_f = torch.zeros(nC, row, col + (nC - 1) * len_shift, device=f.device, dtype=f.dtype)
    for c in range(nC):
        shift_f[c, :, c * len_shift:c * len_shift + col] = f[c, :, :]
    return shift_f


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist


def prepare_data(file_list, file_num):
    #HSI = np.zeros((((512, 512, 28, file_num))))
    HSI = np.zeros((((1024, 1024, 28, file_num))))
    for idx in range(file_num):
        path1 = file_list[idx]
        data = sio.loadmat(path1)
        HSI[:, :, :, idx] = data['img_expand'] / 65535.0
        #HSI[:, :, :, idx] = data['data_slice'] / 65535.0
    HSI[HSI < 0.] = 0.
    HSI[HSI > 1.] = 1.
    return HSI


def prepare_indian_pines(path, n_bands=28):
    """Fallback cube when CAVE .mat files are not on disk. Returns (H, W, C, N)."""
    d = sio.loadmat(path)
    cube = [v for k, v in d.items() if not k.startswith('__')][0].astype(np.float32)
    nb = cube.shape[2] // n_bands
    cube = cube[:, :, :nb * n_bands].reshape(cube.shape[0], cube.shape[1], n_bands, nb).mean(-1)
    cube = cube / max(float(np.percentile(cube, 99.9)), 1e-6)
    cube = np.clip(cube, 0.0, 1.0)
    scenes = [cube, cube[:, ::-1].copy(), cube[::-1, :].copy(), np.rot90(cube, 1).copy()]
    return np.stack(scenes, axis=-1)


def cave_files_exist(file_list):
    return bool(file_list) and os.path.isfile(file_list[0])


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pkl'))
    # file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # result = re.findall(".*model_(.*).pth.*", file_)
            result = re.findall(".*model_(.*).pkl.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    if err < 1.0e-10:
        return 100
    else:
        return 10 * np.log10((data_range ** 2) / err)

def compare_sam(im_true, im_test, eps=1e-8):
    """Mean spectral angle (degrees) over batch and pixels. Arrays are (N,C,H,W)."""
    im_true, im_test = _as_floats(im_true, im_test)
    a = im_true.reshape(im_true.shape[0], im_true.shape[1], -1)
    b = im_test.reshape(im_test.shape[0], im_test.shape[1], -1)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    ang = np.arccos(np.clip(num / den, -1.0, 1.0))
    return float(np.mean(ang) * 180.0 / np.pi)


def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2



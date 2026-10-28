"""Dataset_nb: Dataset.py generalised to any band count and any scene size.

Differences from Dataset.py (behaviour for CAVE 28 is otherwise identical):
  * `HSI` is a list of (H, W, nC) arrays (uint16 /65535 or float32 in [0, 1]); scenes may differ in size.
  * the mosaic branch (`arguement_2`) uses opt.bands / opt.size instead of 28 / 256 / 128.
  * the mask file may hold a 2-D (H, W) mask or a 3-D (H, W, nC') mask; it is repeated / cut to nC.
  * test scenes are used whole (they must already be size x size).
"""
import random

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as tud

from Dataset import arguement_1
from Utils import shift_3


def load_mask(mask_path, nC, size):
    m = sio.loadmat(mask_path)['mask'].astype(np.float32)
    if m.ndim == 2:
        m = np.repeat(m[:, :, None], nC, axis=2)
    elif m.shape[2] < nC:
        m = np.repeat(m[:, :, :1], nC, axis=2)
    else:
        m = m[:, :, :nC]
    assert m.shape[0] >= size and m.shape[1] >= size, "mask is %s, patch is %d" % (m.shape, size)
    return m[:size, :size]  # a larger mask is cropped (train at 128 with the top-left of the 256 mask)


class dataset_nb(tud.Dataset):
    def __init__(self, opt, HSI):
        super().__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        self.len_shift = opt.len_shift
        self.nC = opt.bands
        self.num = opt.trainset_num if self.isTrain else opt.testset_num
        self.HSI = HSI
        self.scene_num = len(HSI)
        mask = load_mask(opt.mask_path, self.nC, self.size)
        self.mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).cuda().float()

    def _random_crop(self, size):
        hsi = self.HSI[random.randint(0, self.scene_num - 1)]
        h, w, _ = hsi.shape
        px = random.randint(0, h - size)
        py = random.randint(0, w - size)
        return to_float(hsi[px:px + size, py:py + size, :])

    def __getitem__(self, index):
        if self.isTrain:
            arg1 = random.randint(1, 2)
            temp = torch.zeros((self.nC, self.size, self.size)).cuda()
            for _ in range(arg1):
                if random.randint(1, 2) == 1:
                    label = torch.from_numpy(np.transpose(self._random_crop(self.size), (2, 0, 1))).cuda().float()
                    label = arguement_1(label)
                else:
                    half = self.size // 2
                    label = torch.zeros((self.nC, self.size, self.size)).cuda()
                    for i in range(4):
                        crop = torch.from_numpy(np.transpose(self._random_crop(half), (2, 0, 1))).cuda().float()
                        r, c = (i // 2) * half, (i % 2) * half
                        label[:, r:r + half, c:c + half] = crop
                temp = label if arg1 == 1 else temp + label / 2
            label = temp
        else:
            hsi = self.HSI[index]
            label = torch.from_numpy(np.transpose(to_float(hsi[:self.size, :self.size, :]), (2, 0, 1))).cuda().float()

        mask = self.mask
        g = torch.sum(shift_3(mask * label, len_shift=self.len_shift), 0)
        Phi_s = torch.sum(shift_3(mask, self.len_shift) ** 2, 0)
        Phi_s[Phi_s == 0] = 1
        return g.unsqueeze(0), label, mask, Phi_s.unsqueeze(0)

    def __len__(self):
        return self.num


def to_float(x):
    """uint16 'img_expand' storage -> float32 in [0, 1]; float arrays are only clipped."""
    if x.dtype == np.uint16:
        return x.astype(np.float32) / 65535.0
    return np.clip(x.astype(np.float32), 0., 1.)


def load_scenes(file_list, bands=None):
    """Load .mat scenes (key 'img_expand' uint16 / 65535, or 'img' float in [0, 1]).

    uint16 scenes are kept as uint16 in RAM (half the footprint of float32, which matters when a
    24 GB CAVE training run shares the machine) and converted per crop by `to_float`.
    """
    out = []
    for p in file_list:
        d = sio.loadmat(p)
        x = d['img_expand'] if 'img_expand' in d else d['img']
        if x.dtype != np.uint16:
            x = to_float(x)
        if bands is not None:
            assert x.shape[2] == bands, "%s has %d bands, expected %d" % (p, x.shape[2], bands)
        out.append(np.ascontiguousarray(x))
    return out

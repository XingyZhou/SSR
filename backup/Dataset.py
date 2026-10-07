# [备份注释] 文件名: Dataset.py
# [备份注释] 作用: CAVE/测试集 .mat 数据读取与 Dataset 封装
# [备份注释] 备份来源: 提交 5da19fd 时的原文件, 内容未改动

import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio
from torchvision.transforms import Resize
from torchvision.transforms import RandomRotation
from Utils import *

def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x


def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1], generate_gt.shape[2] * 2, generate_gt.shape[3] * 2
    divid_point_h = h // 2
    divid_point_w = w // 2
    output_img = torch.zeros(c, h, w, device=generate_gt.device, dtype=generate_gt.dtype)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


class dataset(tud.Dataset):
    def __init__(self, opt, HSI):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        self.path = opt.data_path
        self.scene_num = opt.scene_num
        self.len_shift = opt.len_shift
        self.nC = opt.bands
        if self.isTrain == True:
            self.num = opt.trainset_num
        else:
            self.num = opt.testset_num
        self.HSI = HSI
        self.device = get_device()

        ## load mask (crop to the training patch size)
        data = sio.loadmat(opt.mask_path)
        mask = data['mask']
        if mask.shape[0] < self.size or mask.shape[1] < self.size:
            raise ValueError('mask %s is smaller than --size %d' % (mask.shape, self.size))
        self.mask = mask[:self.size, :self.size]

    def __getitem__(self, index):
        if self.isTrain:
            arg1 = random.randint(1, 2)
            temp = torch.zeros((self.nC, self.size, self.size), device=self.device)
            for i in range(arg1):
                arg2 = random.randint(1, 2)
                if arg2 == 1 or self.size != 256:
                    index1 = random.randint(0, self.scene_num - 1)
                    hsi = self.HSI[:, :, :, index1]
                    shape = np.shape(hsi)
                    px = random.randint(0, shape[0] - self.size)
                    py = random.randint(0, shape[1] - self.size)
                    label = hsi[px:px + self.size:1, py:py + self.size:1, :]
                    label = torch.from_numpy(np.transpose(label, (2, 0, 1))).to(self.device).float()
                    label = arguement_1(label)
                else:
                    processed_data = np.zeros((4, self.size // 2, self.size // 2, 28))
                    sample_list = np.random.randint(0, self.scene_num, 4)
                    h, w, _ = self.HSI[:, :, :, 0].shape
                    for i in range(4):
                        px = random.randint(0, h - self.size // 2)
                        py = random.randint(0, w - self.size // 2)
                        processed_data[i] = self.HSI[:, :, :, sample_list[i]][px:px + self.size // 2,
                                            py:py + self.size // 2, :]
                    label = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2))).to(self.device)
                    label = arguement_2(label)
                if arg1 == 1:
                    temp = label
                else:
                    temp = temp + label / 2
            label = temp
            mask = self.mask
            mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(self.device).float()

        else:
            index1 = index % (1 * 1)
            index2 = index // (1 * 1)
            hsi = self.HSI[:, :, :, index2]
            px = index1 // 1 * self.size
            py = index1 % 1 * self.size
            label = hsi[px:px + self.size:1, py:py + self.size:1, :]
            label = torch.from_numpy(np.transpose(label, (2, 0, 1))).to(self.device).float()
            mask = self.mask
            mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(self.device).float()

        Phi_batch = mask
        g = shift_3(mask * label, len_shift=self.len_shift)
        g = torch.sum(g, 0)
        Phi_s_batch = torch.sum(shift_3(Phi_batch, 2) ** 2, 0)
        Phi_s_batch[Phi_s_batch == 0] = 1

        return g.unsqueeze(0), label, Phi_batch, Phi_s_batch.unsqueeze(0)

    def __len__(self):
        return self.num

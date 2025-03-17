import os
from os import listdir
from os.path import isdir, join
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class RainDrop:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        # 只针对"Sliced"目录下的子文件夹
        sliced_dir = os.path.join(config.data.data_dir, "sliced")
        if not os.path.exists(sliced_dir):
            raise ValueError(f"Sliced directory does not exist: {sliced_dir}")
        # 仅保留目录名称，不重复（排序后更好调试）
        self.list = sorted([folder for folder in os.listdir(sliced_dir)
                            if isdir(join(sliced_dir, folder))])

        # 此处可根据需要拆分训练/测试集，现均使用全部数据
        self.testlist = self.list
        self.trainlist = self.list
        print('-trainlist-', self.trainlist)
        print('-testlist-', self.testlist)

    def get_loaders(self, parse_patches=True, validation='raindrop'):
        print("=> evaluating raindrop test set...")
        train_dataset = RainDropDataset(dir=self.config.data.data_dir,
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=self.trainlist,
                                        parse_patches=parse_patches)

        val_dataset = RainDropDataset(dir=self.config.data.data_dir,
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      filelist=self.testlist,
                                      parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class RainDropDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()
        self.dir = dir
        input_names = []
        gt_names = []
        # filelist 中的每个元素都是"Sliced"下的子目录名，例如 "00001"
        for folder in filelist:
            inpdir = os.path.join(self.dir, 'sliced', folder)
            gtdir = os.path.join(self.dir, 'gt_sliced', folder)
            if not os.path.exists(inpdir) or not os.path.exists(gtdir):
                print(f"Skipping folder {folder} because required subdirectories do not exist")
                continue

            listinpdir = sorted(os.listdir(inpdir))
            listgtdir = sorted(os.listdir(gtdir))
            # 假设输入与GT图像数量一致，一一对应
            for inp_file, gt_file in zip(listinpdir, listgtdir):
                input_names.append(os.path.join(inpdir, inp_file))
                gt_names.append(os.path.join(gtdir, gt_file))

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0] * n, [0] * n, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = [img.crop((y[i], x[i], y[i] + w, x[i] + h)) for i in range(len(x))]
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        normalized_path = os.path.normpath(input_name)
        parts = normalized_path.split(os.sep)
        if len(parts) < 4:
            raise ValueError(f"Invalid path structure: expected at least 4 parts, got {len(parts)} in '{input_name}'")

        datasetname = parts[-4]
        img_vid = parts[-2]
        img_id = os.path.splitext(parts[-1])[0]
        img_id = datasetname + '__' + img_vid + '__' + img_id

        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name)

        if self.parse_patches:
            wd_new = 512
            ht_new = 512
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_crops = self.n_random_crops(input_img, i, j, h, w)
            gt_crops = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_crops[k]), self.transforms(gt_crops[k])], dim=0)
                       for k in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            wd_new = 256
            ht_new = 256
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)

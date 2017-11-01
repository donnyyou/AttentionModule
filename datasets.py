#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import random
from torch.utils import data
from transform import HorizontalFlip, VerticalFlip

import cv2

def default_loader(path):
    return Image.open(path)

class CSDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()

        data_dir = osp.join(root, "cityspace")
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = osp.join(data_dir, "%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "leftImg8bit/" + split + ("/%s_leftImg8bit.png" % name))
                label_file = osp.join(data_dir, "gtFine/" + split + ("/%s_gtFine_labelTrainIds.png" % name))
                if not osp.exists(img_file):
                    img_file = osp.join(data_dir, "leftImg8bit/" + "train_extra" + ("/%s_leftImg8bit.png" % name))
                    label_file = osp.join(data_dir, "gtCoarse/" + "train_extra" + ("/%s_gtCoarse_labelTrainIds.png" % name))

                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        x1 = random.randint(0, 547)
        y1 = random.randint(0, 273)
        x2 = x1 + 1500
        y2 = y1 + 750
        method = random.randint(0, 5)
        crop = random.randint(0, 5)
        if self.split == "val":
            method = 0
            crop = 0

        region = (x1,y1,x2,y2)
        # print region
        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        if crop > 2:
            img = img.crop(region)
        if method == 3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if method == 4:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # print img.size
        # img = img.resize((256, 256), Image.NEAREST)
        # img = np.array(img, dtype=np.uint8)

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        if crop > 2:
            label = label.crop(region)
        if method == 3:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if method == 4:
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        label_size = label.size
        # label image has categorical value, not continuous, so we have to
        # use NEAREST not BILINEAR
        # label = label.resize((256, 256), Image.NEAREST)
        # label = np.array(label, dtype=np.uint8)
        # label[label == 255] = 21

        if self.img_transform is not None:
            img_o = self.img_transform(img)
            # img_h = self.img_transform(self.h_flip(img))
            # img_v = self.img_transform(self.v_flip(img))
            imgs = img_o
        else:
            imgs = img

        if self.label_transform is not None:
            label_o = self.label_transform(label)
            # label_h = self.label_transform(self.h_flip(label))
            # label_v = self.label_transform(self.v_flip(label))
            labels = label_o
        else:
            labels = label

        # print np.array(labels)
        return imgs, labels


class CSTestSet(data.Dataset):
    def __init__(self, root, img_transform=None, label_transform=None):
        self.root = root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = collections.defaultdict(list)

        self.data_dir = osp.join(root, "cityspace")
        self.img_names = os.listdir(osp.join(self.data_dir, "leftImg8bit/all_val"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
        label_name = "_".join(name.split('.')[0].split('_')[:-1]) + "_gtFine_labelTrainIds.png"
        img = Image.open(osp.join(self.data_dir, "leftImg8bit/all_val", name)).convert('RGB')
        label = Image.open(osp.join(self.data_dir, "gtFine/all_val", label_name)).convert('P')
        size = img.size
        # name = name.split(".")[0]

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label, name, size


if __name__ == '__main__':
    dst = CSDataSet("/root/group_incubation_bj")
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = (imgs).numpy()[0] # torchvision.utils.make_grid(imgs).numpy()
            # img = torchvision.utils.make_grid(imgs).numpy()
            print img.shape
            # cv2.imshow("main", img)
            # cv2.waitKey()
            # img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, ::-1]
            print img.shape
            plt.imshow(img.squeeze())
            plt.show()

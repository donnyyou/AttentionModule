#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
from model import FCN 
from datasets import CSDataSet
from loss import CrossEntropy2d, CrossEntropyLoss2d
from transform import ReLabel, ToLabel, ToSP, Scale, Augment, CocoLabel, ReLabel
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from PIL import Image
import numpy as np

import utils
from image_augmentor import ImageAugmentor

image_augmentor = ImageAugmentor()

NUM_CLASSES = 6
MODEL_NAME = "seg-model"

input_transform = Compose([
    Scale((512, 256), Image.BILINEAR),
    Augment(0, image_augmentor),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])
target_transform = Compose([
    Scale((512, 256), Image.NEAREST),
    ToLabel(),
    ReLabel(),
])

trainloader = data.DataLoader(CSDataSet("/root/group-incubation-bj", split="train",
                                        img_transform=input_transform, label_transform=target_transform),
                                        batch_size=10, shuffle=True, pin_memory=True)

valloader = data.DataLoader(CSDataSet("/root/group-incubation-bj", split="val",
                                      img_transform=input_transform, label_transform=target_transform),
                                      batch_size=1, pin_memory=True)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(FCN(NUM_CLASSES))
    model.cuda()

epoches = 8
lr = 1e-3

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

# pretrained_dict = torch.load("./pth/fcn-deconv-40.pth")
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# model.load_state_dict(torch.load("./pth/seg-norm-2.pth"))

model.train()

x_index = 1

for epoch in range(epoches):
    # epoch = epoch_ + 2
    running_loss = 0.0
    iter_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(image)
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        iter_loss += loss.data[0]
        if (i + 1) % 100 == 0:
            print("Iter [%d] Loss: %.4f" % (i+1, iter_loss/100.0))
            iter_loss = 0.0

        if (i + 1) % 300 == 0:
            utils.plot(MODEL_NAME + "-train_loss", x_index, running_loss/300.0)
            print("Epoch [%d] Loss: %.4f" % (x_index, running_loss/300.0))
            running_loss = 0

            val_loss = 0.0
            for j, (images, labels) in enumerate(valloader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(image)
                    labels = Variable(labels)
        
                outputs = model(images)
                loss = criterion(outputs, labels)
        
                val_loss += loss.data[0]
        
            print("Val [%d] Loss: %.4f" % (x_index, val_loss/len(valloader)))
            utils.plot(MODEL_NAME + "-val_loss", x_index, val_loss/len(valloader))
            x_index += 1
            val_loss = 0

    if (epoch+1) % 1 == 0:
        if (epoch + 1) % 3 == 0:
            lr /= 10.0
    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

        torch.save(model.state_dict(), "./pth/" + MODEL_NAME + ("-%d.pth" % (epoch+1)))


torch.save(model.state_dict(), "./pth/" + MODEL_NAME + ".pth")

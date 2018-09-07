#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Alejandro Debus"
__email__ = "aledebus@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F

class DCAE(nn.Module):

    def __init__(self):
        super(DCAE, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 512, kernel_size = 5, stride = 1, padding = 0)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)

        self.unpool4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)

        self.fc6 = nn.Linear(512, 512)

        self.deconv5 = nn.ConvTranspose2d(512, 128, kernel_size = 5, stride = 1, padding = 0)
        self.deconv4_1 = nn.ConvTranspose2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_1 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_2 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2_1 = nn.ConvTranspose2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2_2 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 1, padding = 1)
        self.deconv1_1 = nn.ConvTranspose2d(16, 16, kernel_size = 3, stride = 1, padding = 1)
        self.deconv1_2 = nn.ConvTranspose2d(16, 16, kernel_size = 3, stride = 1, padding = 1)

        self.dconv_rec = nn.ConvTranspose2d(16, 1, kernel_size = 1, stride = 1, padding = 0)

        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.batch_norm5 = nn.BatchNorm2d(512)

    def forward(self, x):

        batch_size_x = x.size(0)

        x = self.conv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x, idxs1 = self.pool1(x)

        x = self.conv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x, idxs2 = self.pool2(x)

        x = self.conv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x, idxs3 = self.pool3(x)

        x = self.conv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs4 = self.pool4(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)

        # --- LINEAR ---
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = x.view(batch_size_x, 512, 1, 1)

        x = self.deconv5(x)
        x = self.batch_norm4(x)
        x = F.relu(x)

        x = self.unpool4(x, idxs4)
        x = self.deconv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)

        x = self.unpool3(x, idxs3)
        x = self.deconv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.unpool2(x, idxs2)
        x = self.deconv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.unpool1(x, idxs1)
        x = self.deconv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.deconv1_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        w = x
        img_rec = self.dconv_rec(x)

        return img_rec, w

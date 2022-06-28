import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class NeuralNet(nn.Module):
    '''
    NN Architecture:
        2D Convolution-Deconvolution Neural Network: CFDNet

        3 Convolutions and 3 Deconvolutions
        PRelu, tanh activation functions
    '''

    def __init__(self):
        super(NeuralNet, self).__init__()
        # CONVOLUTION
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2, 2), stride=(2, 2),
                               padding=0)  # Output size: 3 x(50,30) 64,32

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(2, 2), stride=(2, 2),
                               padding=0)  # Output size: 32 x (32,16)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(2, 2),
                               padding=(0, 0))  # Output size: 256 x (11,5)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5,11), stride=(1, 1),
                               padding=0)  # Output size: 512 x (1,1)

        # DECONVOLUTION
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(5,11), stride=(1, 1),
                                          padding=0)  # THIS#Output size: 512 x (1,1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2,2),
                                          padding=(0, 0))
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 2), stride=(2, 2),
                                          padding=0)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.dropout = nn.Dropout(p=0.02)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.activ = nn.ReLU()
        self.sigmoid = nn.Softmax(dim=0)
        self.batch32 = nn.BatchNorm2d(32)
        self.batch128 = nn.BatchNorm2d(128)
        self.batch256 = nn.BatchNorm2d(256)
        self.batch512 = nn.BatchNorm2d(512)
        self.batchd32 = nn.BatchNorm2d(32)
        self.batchd128 = nn.BatchNorm2d(128)
        self.batchd256 = nn.BatchNorm2d(256)
        self.batchd3 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.conv22 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, x):
        x_in = x[:, 1:4, :, :]

        encode1 = self.conv2(x_in)
        print(np.shape(encode1))

        encode1 = func.tanh(self.batch32(encode1))
        encode2 = self.dropout(func.leaky_relu(self.batch128(self.conv3(encode1)), negative_slope=1e-2, inplace=False))
        print(np.shape(encode2))
        encode3 = torch.tanh(self.batch256(self.conv4(encode2)))
        encode5 = torch.tanh(self.batch512(self.conv6(encode3)))
        print(np.shape(encode5))

        decode1 = torch.tanh(self.deconv1(encode5))
        decode3 = torch.tanh(self.deconv3(decode1))
        decode4 = torch.tanh(self.deconv4(decode3))
        print(np.shape(decode4))
        decode6 = func.leaky_relu(self.deconv5(decode4), negative_slope=1e-2, inplace=False)
        print(np.shape(decode6))

        decode6[:, 0, :, :] = decode6[:, 0, :, :] * x[:, 0, :, :]
        decode6[:, 1, :, :] = decode6[:, 1, :, :] * x[:, 0, :, :]
        decode6[:, 2, :, :] = decode6[:, 2, :, :] * x[:, 0, :, :]

        return decode6

    # backpropagation function
    def backprop(self, x, y, loss, epoch, optimizer):
        torch.autograd.set_detect_anomaly(True)
        self.train()
        inputs = torch.from_numpy(x)
        targets = torch.from_numpy(y[:, 0:3, :, :])
        lossl1 = torch.nn.MSELoss()
        outputs = self(inputs)
        print("AUTOENCODE", np.shape(self.forward(inputs)), np.shape(targets))
        pred = self.forward(inputs)
        obj_val = loss(pred, targets) \
                  #+ loss(pred[:, :, 20:45, 20:45], targets[:, :, 20:45, 20:45])
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()

        return obj_val.item()

    # test function, avoids calculation of gradients
    def test(self, x, y, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(x)
            pred = self.forward(inputs)
            lossl1 = torch.nn.L1Loss()
            # inputs - inputs.reshape(len(inputs),1,5)
            targets = torch.from_numpy(y[:, 0:3, :, :])
            # targets = targets.reshape(len(targets), 5)
            outputs = self(inputs)
            cross_val = loss(pred, targets) \
                        #+ loss(pred[:, :, 20:45, 20:45], targets[:, :, 20:45, 20:45])
        return cross_val.item()
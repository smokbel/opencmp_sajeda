import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import math


class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        # initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(1, channel_size, output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        b, ch, r, c = x.size()
        print(np.shape(self.w), np.shape(x))
        return ((x * self.w).sum(-1) + self.b).view(b, ch, 1, -1)


class NeuralNet(nn.Module):
    '''
    NN Architecture:
        2D Convolution-Deconvolution Neural Network: CFDNet

        3 Convolutions and 3 Deconvolutions
        PRelu, tanh activation functions
    '''


    def __init__(self):
        super(NeuralNet, self).__init__()
        # FC LAYERS
        # self.fc1 = nn.Linear(4, 4*2)
        # self.fc2 = nn.Linear(4*2, 4*4)
        # self.fc3 = nn.Linear(4*4, 4*8)
        # self.fc4= nn.Linear(4*8, 4*4)
        # self.fc5 = nn.Linear(4*4, 4)
        # self.fc6 = nn.Linear(4, 2)

        # CONVOLUTION

        self.fc1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=2, stride=2)

        self.fc2 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=4, stride=4,
                               padding=0)  # Output size: 32 x 2 x (2778,1)
        self.fc3 = nn.Conv1d(in_channels=32, out_channels=512, kernel_size=2, stride=2,
                               padding=0)  # Output size: 256 x 2 x (1389,1)

        self.fc4 = nn.ConvTranspose1d(in_channels=512, out_channels=32, kernel_size=2, stride=2,
                               padding=0)  # Output size:16 x 2 x (5556,1)

        # DECONVOLUTION
        self.fc5 = nn.ConvTranspose1d(in_channels=32, out_channels=4, kernel_size=4, stride=4,
                                          padding=0)  # THIS#Output size: 512 x (1,1)
        self.fc6 = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=2, stride=2,
                                          padding=0)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2, 2), stride=(2, 2),
                                          padding=0)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.dropout = nn.Dropout(p=0.02)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.activ = nn.ReLU()
        self.prelu1 = nn.PReLU(4)
        self.prelu2 = nn.PReLU(2)
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
        x_in = x[:, 0:4, :]
        #x_in = x_in.view(22224, 4)
        print("SHAPE: ", np.shape(x_in))
        layer1 = self.prelu1(self.fc1(x_in))
        print("SHAPE after layer1: ", np.shape(layer1))
        layer2 = self.dropout(func.tanh(self.fc2(layer1)))
        print("SHAPE after layer2: ", np.shape(layer2))
        layer3 = func.tanh(self.fc3(layer2))
        print("SHAPE after layer3: ",np.shape(layer3))
        layer4 = func.tanh(self.fc4(layer3))
        print("SHAPE after layer4: ", np.shape(layer4))
        layer5 = func.tanh(self.fc5(layer4))
        print("SHAPE after layer5: ", np.shape(layer5))
        layer6 = self.prelu2(self.fc6(layer5))
        print("SHAPE after layer6: ", np.shape(layer6))

        return layer6

    # backpropagation function
    def backprop(self, x, y, loss, epoch, optimizer):
        torch.autograd.set_detect_anomaly(True)
        self.train()
        # inputs = torch.from_numpy(x)
        # targets = torch.from_numpy(y)
        #inputs = inputs.to(device)
        lossl1 = torch.nn.MSELoss()
        outputs = self(x)
        pred = self.forward(x)
        print("Prediction: ", pred[0,0,45:55])
        print("Targets: ", y[0,0,45:55])
        obj_val = loss(pred, y) \
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
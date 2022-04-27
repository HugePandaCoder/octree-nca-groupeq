import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.CAModel import CAModel

class CAModel_Deeper(nn.Module):
    """.. note:: Test purposes"""
    def __init__(self, channel_n, fire_rate, device, checkCellsAlive=False, hidden_size=32):
        super(CAModel_Deeper, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.checkCellsAlive = checkCellsAlive

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc4 = nn.Linear(hidden_size, channel_n, bias=False)
        #self.fc3 = nn.Linear(hidden_size, channel_n, bias=False)

        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc2.weight.zero_()

        torch.nn.init.xavier_uniform(self.fc0.weight)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        torch.nn.init.xavier_uniform(self.fc4.weight)

        self.fire_rate = fire_rate

        self.die_after_choice = False
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def choiceMade(self, x):
        return x > 0.5 # F.max_pool2d(x[:, 3, :, :], kernel_size=3, stride=1, padding=1)

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate, angle):
        x = x_in.transpose(1,3)

        if self.checkCellsAlive:
            pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        dx = F.relu(dx)
        dx = self.fc2(dx)
        dx = F.relu(dx)
        dx = self.fc3(dx)
        dx = F.relu(dx)
        dx = self.fc4(dx)


        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        if self.checkCellsAlive:
            #pre_life_mask = self.alive(x)
            post_life_mask = self.alive(x)
            life_mask = (pre_life_mask & post_life_mask).float()
            x = x * life_mask

        x = x.transpose(1,3)

        if self.die_after_choice:
            dead_mask = x_in[..., 3] > 0 #x[..., 3]
            x[..., 3][dead_mask] = 1 #x_in[..., 3][dead_mask]
            #x[..., 3][dead_mask] = 1

        return x #.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x[...,3:] = self.update(x, fire_rate, angle)[...,3:]
        return x
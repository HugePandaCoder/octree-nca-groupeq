import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
 
class BasicNCA3D(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, init_method="standard"):
        super(BasicNCA3D, self).__init__()

        self.device = device
        self.channel_n = channel_n

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x, angle):
        y1 = self.p0(x)
        #y2 = self.p1(x)
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, fire_rate, angle):
        x = x_in.transpose(1,4)
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x

    def forward(self, x, steps=10, fire_rate=0.5, angle=0.0):
        for step in range(steps):
            x2 = self.update(x, fire_rate, angle).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:1], x2[...,1:]), 4)
        return x

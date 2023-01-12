import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
 
class BasicNCA3D_Public(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, init_method="standard"):
        super(BasicNCA3D_Public, self).__init__()

        self.device = device
        self.channel_n = channel_n

        self.world_c_size = 4
        #self.world_communication = torch.zeros((self.world_c_size), dtype=torch.float32, device=self.device, requires_grad=True) #torch.tensor([0, 1, 2, 3])# 

        # One Input
        self.fc0 = nn.Linear(channel_n*2 + self.world_c_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n + self.world_c_size, bias=False)
        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x, world_kn):
        y1 = self.p0(x)
        #y2 = self.p1(x)
        y = torch.cat((x, y1, world_kn.transpose(1,4)),1)
        return y

    def update(self, x_in, fire_rate, world_kn):
        #print("BEGIN UPDATE")
        x = x_in.transpose(1,4)
        #print(x.shape)
        dx = self.perceive(x, world_kn)
        #print(dx.shape)
        dx = dx.transpose(1,4)
        dx = self.fc0(dx)
        dx = F.relu(dx, inplace=False)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)[:, :self.channel_n, :, :, :]

        x = x.transpose(1,4)

        #print(dx.transpose(1,4)[:, self.channel_n:, :, :, :].shape)
        #print(torch.mean(dx.transpose(1,4)[:, self.channel_n:, :, :, :], dim=(2, 3, 4)).shape)
        #print("________________________")
        #print(world_kn.shape)
        world_kn = torch.mean(dx[:, :, :, :, self.channel_n:], dim=(1, 2, 3), keepdim=True)#.transpose(1,4)
        #print(world_kn.shape)
        #print(dx.shape)
        world_kn = world_kn.repeat(1, dx.shape[1], dx.shape[2], dx.shape[3], 1)
        #print(world_kn.shape)



        return x, world_kn

    def forward(self, x, steps=10, fire_rate=0.5, angle=0.0):
        #world_communication = torch.zeros((self.world_c_size), dtype=torch.float32, device=self.device, requires_grad=True) #torch.tensor([0, 1, 2, 3])
        #world_kn = world_communication.expand((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.world_c_size)) #, dtype=torch.float32, device=self.device
        world_kn = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.world_c_size), dtype=torch.float32, device=self.device)
        for step in range(steps):
            #print(world_kn)
            x_back, world_kn = self.update(x, fire_rate, world_kn) #[...,3:][...,3:]
            x2 = x_back.clone()
            x = torch.concat((x[...,0:1], x2[...,1:]), 4)
        return x

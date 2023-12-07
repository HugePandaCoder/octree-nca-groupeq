import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model_BasicNCA3D import BasicNCA3D

class GenNCA_V3_NoHyper(nn.Module):
    r"""Basic implementation of an NCA using a sobel x and y filter for the perception
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", extra_channels=8, kernel_size=7, groups=False, batch_size = 8):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
        """
        super(GenNCA_V3_NoHyper, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.extra_channels = extra_channels
        self.batch_size = batch_size
        self.kernel_size = kernel_size

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

        self.list_backpropTrick = []
        for i in range(batch_size):
            self.list_backpropTrick.append(nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels))

        #with torch.no_grad():
        #    self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y2 = self.p1(x)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """

        x = x_in.transpose(1,3)

        dx = self.perceive(x)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    def forward(self, x, x_vec_in, steps=64, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        x_vec_in = x_vec_in.to(self.device)[:, :, None]
        batch_emb = []
        for i in range(x_vec_in.shape[0]):
            batch_emb.append(self.list_backpropTrick[i].to(self.device)(x_vec_in[i:i+1]))
        #emb = torch.stack(batch_emb, dim=0)
        emb = torch.squeeze(torch.stack(batch_emb, dim=0))

        if x_vec_in.shape[0] == 1:
            emb = emb.to(self.device)[None, :]

        emb = emb.unsqueeze(-2).unsqueeze(-2).repeat(1, x.shape[1], x.shape[2], 1)

        x[..., -6:x.shape[3]] = emb

        
        for step in range(steps):
            x = self.update(x, fire_rate)#.clone() #[...,3:][...,3:]
        return x

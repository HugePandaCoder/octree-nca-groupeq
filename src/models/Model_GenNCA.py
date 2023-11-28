import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.Model_BasicNCA3D import BasicNCA3D
 
class GenNCA(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, extra_channels=8):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
                kernel_size: defines kernel input size
                groups: if channels in input should be interconnected
        """
        super(GenNCA, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        # This is the embeddding for the information
        self.extra_channels = extra_channels

        self.embedding_backpropTrick = nn.Conv3d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels)
        self.embedding = nn.Sequential(
            nn.Conv3d(extra_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv3d(64, extra_channels, kernel_size=1, stride=1, padding=0)
        )

        # One Input
        self.fc0 = nn.Linear(channel_n*2 + extra_channels, hidden_size)
        self.fc1 = nn.Linear(hidden_size + extra_channels, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        self.bn = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False)

        # We need input here that can be clustered afterwards 
        # Its fed into each NCA

        # SET BACKPROP TRICK
        
        with torch.no_grad():
            self.embedding_backpropTrick.weight[self.embedding_backpropTrick.weight != 1] = 1.0
            self.fc1.weight.zero_()


        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, x_vec_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,4)
        dx = self.perceive(x)
        dx = dx.transpose(1,4)
        # <<<---- here vector needs to be converted to image size -> look diffusion

        x_vec_in = x_vec_in.view(x_vec_in.shape[0], 1, 1, 1, x_vec_in.shape[1])
        x_vec_in = x_vec_in.expand(-1, dx.shape[1], dx.shape[2], dx.shape[3], -1)

        emb = self.embedding_backpropTrick(x_vec_in.transpose(1,4))
        emb = self.embedding(emb).transpose(1,4)

        dx = torch.cat((dx, emb), dim=4)
        dx = self.fc0(dx)
        dx = dx.transpose(1,4)
        dx = self.bn(dx)
        dx = dx.transpose(1,4)
        dx = F.relu(dx)
        dx = torch.cat((dx, emb), dim=4)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,4)

        x = x.transpose(1,4)

        return x

    def forward(self, x, x_vec_in, steps=10, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        x_vec_in = x_vec_in.to(self.device)
        for step in range(steps):
            x2 = self.update(x, x_vec_in, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x

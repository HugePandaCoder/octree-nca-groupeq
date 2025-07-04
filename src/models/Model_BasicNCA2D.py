import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BasicNCA2D(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False,
                 normalization="batch"):
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
        super(BasicNCA2D, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        
        if normalization == "batch":
            self.bn = torch.nn.BatchNorm2d(hidden_size, track_running_stats=False)
        elif normalization == "layer":
            self.bn = torch.nn.LayerNorm(hidden_size)
        elif normalization == "group":
            self.bn = torch.nn.GroupNorm(1, hidden_size)
        elif normalization == "none":
            self.bn = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type {normalization}")


        
        with torch.no_grad():
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
        dx = dx.transpose(1,3)
        dx = self.bn(dx)
        dx = dx.transpose(1,3)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    def forward(self, x, steps=10, fire_rate=0.5, visualize: bool=False):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        if visualize:
            gallery = []
        #x: BHWC
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            #x2: BHWC
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 3)
            if visualize:
                gallery.append(x.detach().cpu())
        
        if visualize:
            return x, gallery
        return x

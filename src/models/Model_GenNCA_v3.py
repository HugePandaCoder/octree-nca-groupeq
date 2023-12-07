import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.Model_BasicNCA3D import BasicNCA3D

class HyperNetwork(nn.Module):
    def __init__(self, input_size, channel_n, kernel_size, hidden_size):
        super(HyperNetwork, self).__init__()
        self.conv3d = channel_n * kernel_size * kernel_size
        self.fc0 = (channel_n*2) * hidden_size
        self.fc1 = (hidden_size) * channel_n
        output_size =  self.conv3d + self.fc0 + self.fc1
            
        #self.fc = nn.Linear(input_size, output_size)
        if False:
            self.fc = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, output_size)
            )
        self.lin01 = nn.Linear(input_size, 64)
        self.lin02 = nn.Linear(64+input_size, 512)
        #self.lin03 = nn.Linear(64+input_size, 256)
        #self.lin04 = nn.Linear(256+input_size, 4096)
        self.lin05 = nn.Linear(512+input_size, output_size)
        self.silu = nn.ReLU()
        
    def forward(self, x):
        # Generate flattened weights
        dx = self.silu(self.lin01(x))
        dx = torch.cat((dx, x), dim=1)
        dx = self.silu(self.lin02(dx))
        dx = torch.cat((dx, x), dim=1)
        #dx = self.silu(self.lin03(dx))
        #dx = torch.cat((dx, x), dim=1)
        #dx = self.silu(self.lin04(dx))
        #dx = torch.cat((dx, x), dim=1)
        weights = self.lin05(dx)
        return weights

class GenNCA_v3(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False, extra_channels=8, batch_size = 8):
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
        super(GenNCA_v3, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size

        # This is the embeddding for the information
        self.extra_channels = extra_channels

        self.list_backpropTrick = []
        for i in range(batch_size):
            self.list_backpropTrick.append(nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels))

        #self.embedding_backpropTrick = nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels)
        #self.embedding = nn.Sequential(
        #    nn.Conv3d(extra_channels, 64, kernel_size=1, stride=1, padding=0),
        #    nn.SiLU(),
        #    nn.Conv3d(64, extra_channels, kernel_size=1, stride=1, padding=0)
        #)

        # One Input
        #self.fc0 = nn.Linear(channel_n*2 + extra_channels, hidden_size)
        #self.fc1 = nn.Linear(hidden_size + extra_channels, channel_n, bias=False)
        self.padding = int((kernel_size-1) / 2)

        #self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=self.padding, padding_mode="reflect", groups=channel_n)
        self.bn = torch.nn.BatchNorm2d(hidden_size, track_running_stats=False)

        self.hypernetwork = HyperNetwork(extra_channels, channel_n, kernel_size, hidden_size)

        # We need input here that can be clustered afterwards 
        # Its fed into each NCA

        # SET BACKPROP TRICK
        
        with torch.no_grad():
            for i in range(batch_size):
                self.list_backpropTrick[i].weight[self.list_backpropTrick[i].weight != 1] = 1.0
            #self.embedding_backpropTrick.weight[self.embedding_backpropTrick.weight != 1] = 1.0
            #self.fc1.weight.zero_()


        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x, generated_weights):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        #y1 = generated_weights(x)
        batch_size = generated_weights.shape[0]
        output = []
        for i in range(batch_size):
            weights = generated_weights[i, 0:self.hypernetwork.conv3d].view(self.channel_n, 1, self.kernel_size, self.kernel_size)
            output.append(F.conv2d(x[i:i+1], weights, padding=self.padding, groups=self.channel_n))
        y1 = torch.cat(output, dim=0)
        y = torch.cat((x,y1),1)
        return y

    def update(self, x_in, x_vec_in, fire_rate, generated_weights):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        x = x_in.transpose(1,3)
        dx = self.perceive(x, generated_weights)
        dx = dx.transpose(1,3)
        batch_size = x_in.shape[0]
        # <<<---- here vector needs to be converted to image size -> look diffusion

        #x_vec_in = x_vec_in.view(x_vec_in.shape[0], 1, 1, 1, x_vec_in.shape[1])
        #x_vec_in = x_vec_in.expand(-1, dx.shape[1], dx.shape[2], dx.shape[3], -1)

        #emb = self.embedding_backpropTrick(x_vec_in.transpose(1,4))
        #emb = self.embedding(emb).transpose(1,4)

        #dx = torch.cat((dx, emb), dim=4)
        #dx = self.fc0(dx)

        dx = dx.transpose(1,3)
        output = []
        for i in range(batch_size):
            weights = generated_weights[i, self.hypernetwork.conv3d:self.hypernetwork.conv3d+self.hypernetwork.fc0].view(self.hidden_size, self.channel_n*2, 1, 1)
            #weights = generated_weights[i, self.hypernetwork.conv3d:self.hypernetwork.conv3d+self.hypernetwork.fc0].view(self.channel_n*2, self.hidden_size)
            output.append(F.conv2d(dx[i:i+1], weights, padding=0))
        dx = torch.cat(output, dim=0)

        dx = self.bn(dx)
        dx = dx.transpose(1,3)
        dx = F.relu(dx)
        #dx = torch.cat((dx, emb), dim=4)
        #dx = self.fc1(dx)

        dx = dx.transpose(1,3)
        output = []
        for i in range(batch_size):
            weights = generated_weights[i, self.hypernetwork.conv3d+self.hypernetwork.fc0:self.hypernetwork.conv3d+self.hypernetwork.fc0+self.hypernetwork.fc1].view(self.channel_n, self.hidden_size, 1, 1)
            #weights = generated_weights[i, self.hypernetwork.conv3d:self.hypernetwork.conv3d+self.hypernetwork.fc0].view(self.channel_n*2, self.hidden_size)
            output.append(F.conv2d(dx[i:i+1], weights, padding=0))
        dx = torch.cat(output, dim=0)
        dx = dx.transpose(1,3)

        #for i in range(batch_size):
        #    weights = generated_weights[i, self.hypernetwork.conv3d+self.hypernetwork.fc0:self.hypernetwork.conv3d+self.hypernetwork.fc0+self.hypernetwork.fc1].view(self.hidden_size, self.channel_n)
        #    dx[i:i+1] = F.linear(dx[i:i+1], weights)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        #stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),dx.size(4)])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        x = x.transpose(1,3)

        return x

    def forward(self, x, x_vec_in, steps=10, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        x_vec_in = x_vec_in.to(self.device)[:, :, None]

        batch_emb = []
        for i in range(x_vec_in.shape[0]):
            batch_emb.append(self.list_backpropTrick[i].to(self.device)(x_vec_in[i]))
        emb = torch.stack(batch_emb, dim=0)
        emb = torch.squeeze(torch.stack(batch_emb, dim=0), dim=2)
        #emb = torch.squeeze(torch.stack(emb, dim=0), dim=2)

        generated_weights = self.hypernetwork(emb)


        output = []
        for step in range(steps):
            x = self.update(x, x_vec_in, fire_rate, generated_weights)
        return x

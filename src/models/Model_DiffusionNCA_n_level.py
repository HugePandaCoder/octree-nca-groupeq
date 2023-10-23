import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
 
class Level(nn.Module):
    def __init__(self, input_channels, channel_n, hidden_size, drop_out_rate, device, kernelSize=3):
        super().__init__() 
        extra_channels = 4

        self.device=device
        self.input_channels = input_channels
        self.channel_n = channel_n
        self.kernelSize = kernelSize

        padding = int((kernelSize-1)/2)

        # ---------------- MODEL 0 -----------------
        self.drop0 = nn.Dropout(drop_out_rate)
        self.normalize = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        self.conv1 = nn.Conv2d(channel_n//3+extra_channels, channel_n//3, kernel_size=self.kernelSize, stride=1, padding=padding, padding_mode="circular")
        self.conv3 = nn.Conv2d(channel_n//3+extra_channels, channel_n//3, kernel_size=self.kernelSize, stride=1, padding=3, padding_mode="circular", dilation=3)
        self.conv7 = nn.Conv2d(channel_n//3+extra_channels, channel_n//3, kernel_size=self.kernelSize, stride=1, padding=7, padding_mode="circular", dilation=7)
        self.fc0 = nn.Conv2d(channel_n*2+extra_channels*3, hidden_size, kernel_size=1, stride=1, padding=0) 
        self.fc1 = nn.Conv2d(hidden_size+extra_channels, channel_n, kernel_size=1, stride=1, padding=0)

        self.embedding = nn.Sequential(
            nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0)
        )

    def channel_embedding_4d(self, inputs: torch.Tensor, max_period: int = 10000):
        """ Sinusoidal channel embeddings for 4D data.

        :param inputs: Input tensor of shape (batch_size, channels, height, width).
        :param max_period: Maximum period for the sinusoidal function.
        :return: Input tensor with channel encodings.
        """

        channels = inputs.shape[1]

        freq = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=channels//2, dtype=torch.float32) / (channels//2)
        ).to(inputs.device)

        channel_steps = torch.arange(channels, device=inputs.device).float()[None, :, None, None]


        for i in range(channels):
            if i == 0:
                out = inputs[:,i:i+1,...] * torch.cat([torch.cos(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None]), 
                                        torch.sin(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None])], dim=1)
            else:
                out_loc = inputs[:,i:i+1,...] * torch.cat([torch.cos(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None]), 
                                        torch.sin(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None])], dim=1)
                out = torch.cat([out, out_loc], dim=1)
        
        #inputs = inputs.repeat(1, 4, 1, 1)
        #print(channel_steps.shape, freq[None, :, None, None].shape, inputs.shape, channel_emb.shape)

        return out #inputs * channel_emb

    def perceive(self, x1, x2, x3):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """

        y1 = self.conv1(x1)
        y2 = self.conv3(x2)
        y3 = self.conv7(x3)
        
        #y2 = conv1(x)
        y = torch.cat((x1, x2, x3, y1, y2, y3),1)#torch.cat((x,y1,y2),1)
        return y

    def step(self, x, fire_rate, alive_rate, step = 0):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """

        dx = x

        if True: # Diagonal
            
            pos_x = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            pos_y = torch.linspace(1, 0, dx.shape[2]).expand(dx.shape[0], 1, dx.shape[3], dx.shape[2]).to(self.device).transpose(2,3)#torch.transpose(alive, 2,3)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
            step = torch.tensor(step).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)

            pos_t_enc = self.channel_embedding_4d(torch.concat((pos_x, pos_y, alive_rate, step), 1))
            pos_t_enc = self.embedding(pos_t_enc)

        dx1 = torch.concat((dx[:, 0:(dx.shape[1]//3), ...], pos_t_enc), 1)
        dx2 = torch.concat((dx[:, (dx.shape[1]//3):2*(dx.shape[1]//3), ...], pos_t_enc), 1)
        dx3 = torch.concat((dx[:, 2*(dx.shape[1]//3):, ...], pos_t_enc), 1)

        dx = self.perceive(dx1, dx2, dx3)

        dx = self.fc0(dx)


        dx = self.normalize(dx)
        dx = dx.transpose(1, 3)
        dx = F.leaky_relu(dx)

        dx = dx.transpose(1, 3)
        dx = torch.concat((dx, pos_t_enc), 1)
        dx = self.fc1(dx)
        dx = dx.transpose(1, 3)

        if fire_rate is None:
            fire_rate = self.fire_rate

        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device)) > fire_rate
        stochastic = stochastic.float()
        
        dx = dx * stochastic

        dx = dx.transpose(1, 3)

        x = x + dx 
        return x
    
    def forward(self, x_input, steps=10, fire_rate=None, t=0, epoch=0, **kwargs):
        return


class DiffusionNCA_fft2(nn.Module):
    r"""Implementation of Diffusion NCA
    """

    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA_fft2, self).__init__() #  channel_n, fire_rate, device, hidden_size)

        self.input_channels = input_channels

        self.level0 = Level(input_channels, channel_n, hidden_size*60, drop_out_rate, device, kernelSize=3)
        self.level1 = Level(input_channels, channel_n, hidden_size*3, drop_out_rate, device, kernelSize=3)
        self.level2 = Level(input_channels, channel_n, hidden_size, drop_out_rate, device, kernelSize=3)


    
    def forward(self, x_input, steps=10, fire_rate=None, t=0, epoch=0, **kwargs):
        r"""
        forward pass from NCA
        :param x: perception
        :param steps: number of steps, such that far pixel can communicate
        :param fire_rate:
        :param angle: rotation
        :return: updated input
        """

        scale_factor = 2
        level = 2

        # STEP 1
        x = x_input.transpose(1, 3) 
        x_lowres = F.interpolate(x, size=(x.shape[2]//(scale_factor*level), x.shape[3]//(scale_factor*level)), mode='bilinear')
        for step in range(steps):
            x_lowres[:, self.input_channels:, ...] = self.level0.step(x_lowres, fire_rate, alive_rate=t, step=step/steps)[:, self.input_channels:, ...] 
        up = torch.nn.Upsample(scale_factor=scale_factor*level, mode='nearest')
        x_upscaled = up(x_lowres)
        x[:, self.input_channels:, ...] = x_upscaled[:, self.input_channels:, ...]

        # STEP 2
        x = x_input.transpose(1, 3) 
        x_lowres = F.interpolate(x, size=(x.shape[2]//(scale_factor*(level-1)), x.shape[3]//(scale_factor*(level-1))), mode='bilinear')
        for step in range(steps):
            x_lowres[:, self.input_channels:, ...] = self.level1.step(x_lowres, fire_rate, alive_rate=t, step=step/steps)[:, self.input_channels:, ...] 
        up = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
        x_upscaled = up(x_lowres)
        x[:, self.input_channels:, ...] = x_upscaled[:, self.input_channels:, ...]

        # STEP 3
        for step in range(steps):
            x[:, self.input_channels:, ...] = self.level2.step(x, fire_rate, alive_rate=t, step=step/steps)[:, self.input_channels:, ...] 

        x = x.transpose(1, 3)
        
        return x
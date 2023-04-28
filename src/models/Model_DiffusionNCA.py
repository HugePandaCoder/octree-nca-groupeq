import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
 
class DiffusionNCA(BackboneNCA):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm0 = nn.LayerNorm([img_size, img_size, hidden_size])

        # Complex
        self.complex = False
        if self.complex:
            self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
            self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect", dtype=torch.complex64)
            self.fc0 = nn.Linear(channel_n*3, hidden_size, dtype=torch.complex64)
            self.fc1 = nn.Linear(hidden_size, channel_n, bias=False, dtype=torch.complex64)
        #self.p0 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        #self.p1 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def update(self, x_in, fire_rate):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """
        
        x = x_in.transpose(1, 3)

        #if self.complex:
        #    x = torch.fft.fft2(x, norm="forward")


        dx = self.perceive(x)
        dx = dx.transpose(1, 3)

        dx = self.fc0(dx)
        dx = F.leaky_relu(dx)  # .relu(dx)

        dx = self.norm0(dx)
        dx = self.drop0(dx)

        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x + dx.transpose(1, 3)

        #if self.complex:
        #    x = torch.fft.ifft2(x, norm="forward")

        x = x.transpose(1, 3)

        #print("back", x.shape, x.dtype)

        return x
    
    def forward(self, x, steps=1, fire_rate=None, t=0):
        r"""
        forward pass from NCA
        :param x: perception
        :param steps: number of steps, such that far pixel can communicate
        :param fire_rate:
        :param angle: rotation
        :return: updated input
        """
        #print(t)
        x_min, x_max = torch.min(x), torch.max(x)
        abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
        #x = x/abs_max
        
        #x = (x - x_min) / (x_max - x_min) 
        #x = x*2-1

        scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
        scaled_t = scaled_t.transpose(0, 3)
        x[:, :, :, -1:] = scaled_t

        # Add pos
        if True:
            #x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            #y_count = torch.linspace(0, 1, x.shape[2]).expand(x.shape[0], x.shape[1], 1, x.shape[2]).transpose(2,3)
            x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            x_count = (x_count + torch.transpose(x_count, 1,2)) / 2
            #y_count = torch.transpose(x_count, 2, 3)
            x[:, :, :, -2:-1] = x_count
            #x[:, :, :, -3:-2] = y_count

            #x[:, :, :, -6:-3] = x[:, :, :, 0:3]


            #x[:, :, :, 0:-6] = torch.randn_like(x[:, :, :, 0:-6])*0.01
            
            #x[:, :, :, 0:3] = torch.zeros_like(x[:, :, :, 0:3])

        #x = x.type(torch.cfloat)
        #if self.complex:
        #    x = torch.fft.fftn(x)

        for step in range(steps):
            #print(x.shape, scaled_t.shape)
            x_update = self.update(x, fire_rate).clone()
            #x = torch.concat((x_update[..., 0:-6], x[..., -6:]), 3)#x_update
            x = x_update
        #x = x*abs_max
        
        #x = (x+1)/2
        #x = x * (x_max - x_min) + x_min

            #x = torch.concat((x_update[..., :3], x[..., 3:4], x_update[..., 4:]), 3) # Leave 3:4
        #if self.complex:
        #    x = torch.fft.ifftn(x)
        #x = x.type(torch.float)
        return x
import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
 
class DiffusionNCA_fft(nn.Module):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA_fft, self).__init__() #  channel_n, fire_rate, device, hidden_size)

        self.device=device

        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm_real = nn.LayerNorm([img_size, img_size, hidden_size])
        self.norm_imag = nn.LayerNorm([img_size, img_size, hidden_size])

        # Complex
        self.p0_real = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1_real = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.fc0_real = nn.Linear(channel_n*3, hidden_size)
        self.fc1_real = nn.Linear(hidden_size, channel_n, bias=False)

        self.p0_imag = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1_imag = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.fc0_imag = nn.Linear(channel_n*3, hidden_size)
        self.fc1_imag = nn.Linear(hidden_size, channel_n, bias=False)
        #self.p0 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        #self.p1 = torch.conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def perceive_real(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        #print(x.shape)
        y1 = self.p0_real(x)
        y2 = self.p1_real(x)
        y = torch.cat((x,y1,y2),1)
        return y
    
    def perceive_imag(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0_imag(x)
        y2 = self.p1_imag(x)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """
        
        # Convert to Fourier
        #x = torch.fft.fftn(x_in)

        x = x_in.transpose(1, 3)

        x = torch.fft.fft2(x)

        #dx = torch.tensor(x, dtype=torch.complex64)
        dx = x

        dx = torch.complex(self.perceive_real(dx.real), self.perceive_imag(dx.imag))
        #dx2 = torch.tensor(self.perceive_real(dx.real), dtype=torch.complex64, requires_grad=True)
        #dx2 = self.perceive_imag(dx.imag)
        dx = dx.transpose(1, 3)

        dx = torch.complex(self.fc0_real(dx.real), self.fc0_imag(dx.imag))
        dx = torch.complex(F.leaky_relu(dx.real), F.leaky_relu(dx.imag)) # .relu(dx)

        #dx = torch.complex(self.norm_real(dx.real), self.norm_imag(dx.imag))
        dx = torch.complex(self.drop0(dx.real), self.drop0(dx.imag))

        dx = torch.complex(self.fc1_real(dx.real), self.fc1_real(dx.imag))

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x + dx.transpose(1, 3)

        x = torch.fft.ifft2(x)

        x = x.transpose(1, 3)

        #x = torch.fft.ifftn(x_in)

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
        scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
        scaled_t = scaled_t.transpose(0, 3)
        x[:, :, :, -1:] = scaled_t


        # Add pos
        if False:
            x_count = torch.linspace(0, 1, x.shape[1]).expand(x.shape[0], 1, x.shape[2], x.shape[1]).transpose(1,3)
            y_count = torch.linspace(0, 1, x.shape[2]).expand(x.shape[0], x.shape[1], 1, x.shape[2]).transpose(2,3)

            x[:, :, :, -2:-1] = x_count
            x[:, :, :, -3:-2] = y_count
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
            
            #x = torch.concat((x_update[..., :3], x[..., 3:4], x_update[..., 4:]), 3) # Leave 3:4
        #if self.complex:
        #    x = torch.fft.ifftn(x)
        #x = x.type(torch.float)
        return x
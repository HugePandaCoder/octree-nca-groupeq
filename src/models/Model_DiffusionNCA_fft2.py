import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
from matplotlib import pyplot as plt
 
class DiffusionNCA_fft2(nn.Module):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA_fft2, self).__init__() #  channel_n, fire_rate, device, hidden_size)

        self.device=device

        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm_real = nn.LayerNorm([img_size, img_size, hidden_size])

        # Complex
        self.p0_real = nn.Conv2d(channel_n*2, channel_n*2, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1_real = nn.Conv2d(channel_n*2, channel_n*2, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.fc0_real = nn.Linear(channel_n*3*2, hidden_size)
        self.fc1_real = nn.Linear(hidden_size, channel_n*2, bias=False)

        self.bn = nn.BatchNorm2d(hidden_size)

    def perceive(self, x):
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

    def update(self, x, fire_rate, alive_rate):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """
        
        # Convert to Fourier
        #x = x_in.transpose(1, 3)
        #x = torch.fft.fft2(x)

        if True:
            x_count = torch.linspace(0, 1, x.shape[2]).expand(x.shape[0], 1, x.shape[2], x.shape[3])#.transpose(1,3)
            y_count = torch.transpose(x_count, 2, 3)
            #y_count = torch.linspace(0, 1, x.shape[3]).expand(x.shape[0], x.shape[2], 1, x.shape[3])#.transpose(2,3)

            #x[:, -2:-1, :, :] = x_count
            #x[:, -3:-2, :, :] = y_count
            #x[:, -1:, :, :] = self.scaled_t.transpose(1, 3)

            alive = torch.linspace(1, 0, x.shape[3]).expand(x.shape[0], 1, x.shape[2], x.shape[3]).to(self.device)
            #alive = alive * torch.transpose(alive, 2,3)
            alive = (alive + torch.transpose(alive, 2,3)) / 2

        dx = torch.concat((x.real, x.imag), 1)

        dx[:, -1:, :, :] = alive

        #plt.imshow(dx.real[0, 0, :, :].detach().cpu().numpy())
        #plt.show()
        #plt.imshow(dx.real[0, -1, :, :].detach().cpu().numpy())
        #plt.show()

        dx = self.perceive(dx)

        dx = dx.transpose(1, 3)

        dx = self.fc0_real(dx)

        dx = dx.transpose(1, 3)
        #dx = self.bn(dx)
        dx = dx.transpose(1, 3)
        dx = F.leaky_relu(dx)


        dx = self.norm_real(dx)
        dx = self.drop0(dx)

        dx = self.fc1_real(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        #print(dx.shape)
        #print(alive_rate)
        alive_mask = (alive >= alive_rate) & (alive <= (alive_rate + (1/dx.shape[3])*20))
        
        #print(alive.shape)
        #print(alive_mask.shape)
        #plt.imshow(alive_mask[0, 0, :, :].detach().cpu().numpy())
        #plt.show()


        if False:
            dx = dx * alive_mask.transpose(1, 3)
        
        
        #post_life_mask = self.alive(x)

        dx = torch.complex(torch.split(dx, int(dx.shape[3]/2), dim=3)[0], torch.split(dx, int(dx.shape[3]/2), dim=3)[1])
        x = x + dx.transpose(1, 3)
        
        #x = torch.fft.ifft2(x)
        #x = x.transpose(1, 3)


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
        #scaled_t = t.expand(1, x.shape[1], x.shape[2], x.shape[0])#,x.shape[2],1
        #scaled_t = scaled_t.transpose(0, 3)
        #self.scaled_t = scaled_t
        
        #x[:, :, :, -1:] = scaled_t


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

        # Convert to Fourier
        #print(x.shape)

        x = x.transpose(1, 3)
        print("FFT", torch.max(x), torch.min(x))
        x = torch.fft.fft2(x)#, norm="forward")
        #print("FFT_AFTER", torch.max(x.real), torch.min(x.real), torch.max(x.imag), torch.min(x.imag))
        #min_real, max_real = -500, 500 #torch.max(x.real), torch.min(x.real)
        #min_imag, max_imag = -256, 256 #torch.max(x.imag), torch.min(x.imag)
        #x.real = (x.real - min_real) / (max_real - min_real)
        #x.imag = (x.imag - min_imag) / (max_imag - min_imag)

        for step in range(steps):
            #print(x.shape, scaled_t.shape)
            x_update = self.update(x, fire_rate, alive_rate=1-(step/x.shape[2])) #.clone()
            
            #x = torch.concat((x_update[..., 0:-6], x[..., -6:]), 3)#x_update
            x = x_update
        #x.real = x.real * (max_real - min_real) + min_real#(x.real - min_real) / max_real
        #x.imag = x.imag * (max_imag - min_imag) + min_imag
        x = torch.fft.ifft2(x)#, norm="forward")

        x = x.transpose(1, 3)
        #raise Exception("STOP!")



            #x = torch.concat((x_update[..., :3], x[..., 3:4], x_update[..., 4:]), 3) # Leave 3:4
        #if self.complex:
        #    x = torch.fft.ifftn(x)
        #x = x.type(torch.float)

        return x
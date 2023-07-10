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

        extra_channels = 3

        self.device=device


        # ---------------- MODEL 0 -----------------
        self.drop0 = nn.Dropout(drop_out_rate)
        #self.norm_real = nn.LayerNorm([img_size, img_size, hidden_size])

        #self.norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        self.norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=channel_n*2+extra_channels)

        # Complex
        kernelSize = 7

        if kernelSize == 7:
            self.p0_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2+extra_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
            self.p1_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2+extra_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        else:
            self.p0_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2+extra_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
            self.p1_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2+extra_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        
        self.fc0_real = nn.Linear(channel_n*3*2+extra_channels*3, hidden_size)
        self.fc1_real = nn.Linear(hidden_size, channel_n*2, bias=False)

        self.model_0 = {"dropout": self.drop0, "normalisation":self.norm_real2, "conv0": self.p0_real, "conv1": self.p1_real, "fc0": self.fc0_real, "fc1": self.fc1_real}

        if False:
            self.fc_mid = nn.Linear(hidden_size, hidden_size)
            self.norm_mid = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
            self.fc_mid2 = nn.Linear(hidden_size, hidden_size)
            self.norm_mid2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        #self.fc_mid3 = nn.Linear(hidden_size, hidden_size)

        self.bn = nn.BatchNorm2d(hidden_size)

        # In real space
        if True:
            self.real_drop0 = nn.Dropout(drop_out_rate)
            self.real_norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=channel_n+extra_channels)
            if kernelSize == 7:
                self.real_p0_real = nn.Conv2d(channel_n+extra_channels, channel_n+extra_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect", groups=channel_n+extra_channels)#reflect, groups=channel_n*2)
                self.real_p1_real = nn.Conv2d(channel_n+extra_channels, channel_n+extra_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect", groups=channel_n+extra_channels)#, groups=channel_n*2)
            else:
                self.real_p0_real = nn.Conv2d(channel_n+extra_channels, channel_n+extra_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", groups=channel_n+extra_channels)#reflect, groups=channel_n*2)
                self.real_p1_real = nn.Conv2d(channel_n+extra_channels, channel_n+extra_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", groups=channel_n+extra_channels)#, groups=channel_n*2)
            self.real_fc0_real = nn.Linear(channel_n*3+extra_channels*3, hidden_size)
            self.real_fc1_real = nn.Linear(hidden_size, channel_n, bias=False)

            self.model_1 = {"dropout": self.real_drop0, "normalisation":self.real_norm_real2, "conv0": self.real_p0_real, "conv1": self.real_p1_real, "fc0": self.real_fc0_real, "fc1": self.real_fc1_real}

            #self.real

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
    
    def perceive_real(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.real_p0_real(x)
        y2 = self.real_p1_real(x)
        y = torch.cat((x,y1,y2),1)
        return y
    
    def perceive_dict(self, x, conv0, conv1):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = conv0(x)
        y2 = conv1(x)
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
        
        #print("ALIVE_RATE", alive_rate)

        # Convert to Fourier
        #x = x_in.transpose(1, 3)
        #x = torch.fft.fft2(x)

        

        if True: # Diagonal
            alive = torch.linspace(1, 0, x.shape[3]).expand(x.shape[0], 1, x.shape[2], x.shape[3]).to(self.device)
            #alive = (alive + torch.transpose(alive, 2,3)) / 2
            pos_x = alive
            pos_y = torch.transpose(alive, 2,3)
            #print(alive_rate.shape, pos_x.shape)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
        if True: # Square
            x_count = torch.linspace(0, 1, x.shape[2]).expand(x.shape[0], 1, x.shape[2], x.shape[3]).to(self.device)#.transpose(1,3)
            y_count = torch.transpose(x_count, 2, 3)
            #print(x_count.shape(),y_count.shape())
            pos = (x_count + y_count) /2
            alive = torch.maximum(y_count, x_count) #x_count[x_count > y_count] = y_count
            #alive = x_count

            

            #print(alive_rate.shape)

        dx = torch.concat((x.real, x.imag), 1)

        dx = torch.concat((dx, pos_x, pos_y, alive_rate), 1)

        #dx = dx.transpose(1, 3)
        #print("SHAPE", dx.shape)
        dx = self.norm_real2(dx)
       # dx = dx.transpose(1, 3)

        #dx[:, -1:, :, :] = pos_x
        #dx[:, -2:, :, :] = pos_y

        #plt.imshow(dx.real[0, 0, :, :].detach().cpu().numpy())
        #plt.show()
        #exit()
        
        # UNCOMMENT HERE
        #plt.imshow(dx.real[0, -1, :, :].detach().cpu().numpy())
        #plt.show()
        #exit()

        dx = self.perceive(dx)

        

        dx = dx.transpose(1, 3)

        dx = self.fc0_real(dx)

        dx = F.leaky_relu(dx)
        #dx = F.relu(dx)


        #dx = self.norm_real(dx)
        
        #dx = dx.transpose(1, 3)
        #dx = self.norm_real2(dx)
        #dx = dx.transpose(1, 3)

        # Added 
        if False:
            dx = self.fc_mid(dx)
            dx = dx.transpose(1, 3)
            dx = self.norm_mid(dx)
            dx = dx.transpose(1, 3)
            dx = F.leaky_relu(dx)

            dx = self.fc_mid2(dx)
            dx = dx.transpose(1, 3)
            dx = self.norm_mid2(dx)
            dx = dx.transpose(1, 3)
            dx = F.leaky_relu(dx)

        #dx = self.fc_mid2(dx)
        #dx = self.fc_mid3(dx)
        #dx = F.leaky_relu(dx)

        #dx = self.fc_mid2(dx)
        #dx = F.leaky_relu(dx)

        
        
        dx = self.drop0(dx)
        dx = self.fc1_real(dx)

        if False: # basic alive mask
            alive_rate = alive_rate.expand_as(alive.transpose(0, 3))
            alive_rate = alive_rate.transpose(0, 3)
            #print("ALIVE_RATE", alive_rate[...,0], alive[...,0])
            #alive_mask = (alive >= alive_rate) & (alive <= (alive_rate + (1/dx.shape[3])*20))
            #alive_mask = ((1-alive) <= alive_rate) & ((1-alive) >= (alive_rate - (1/dx.shape[3])*10))
            alive_mask = (alive <= 0.2)
            alive_mask[alive_mask == 0] = 0#.1
            # UNCOMMENT HERE
            #plt.imshow(alive_mask[0, 0, :, :].detach().cpu().numpy())
            #plt.show()
            alive_mask = alive_mask.transpose(1, 3)
        if False: # begining start alive mask
            alive_rate = alive_rate.expand_as(alive.transpose(0, 3))
            alive_rate = alive_rate.transpose(0, 3)
            #print("ALIVE", alive.shape, alive_rate.shape)
            alive_mask = ((alive + (1/dx.shape[3])*2) <= alive_rate) & (alive >= (alive_rate - (1/dx.shape[3])*2)) # *10

        #if True:
        #    dx = dx * alive_mask.transpose(1, 3)


        if fire_rate is None:
            fire_rate = self.fire_rate

        #print(torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).shape(), alive_mask.shape() )
        #stochastic = (torch.mul(torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device),alive_mask)) > fire_rate
        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device)) > fire_rate
        stochastic = stochastic.float()
        
        dx = dx * stochastic

        #print(dx.shape)
        #print(alive_rate)


        #if False: # box alive mask



        #print(alive.shape)
        #print(alive_mask.shape)
        
        
        
        #plt.imshow(alive_mask[1, 0, :, :].detach().cpu().numpy())
        #plt.show()
        #plt.imshow(alive[0, 0, :, :].detach().cpu().numpy())
        #plt.show()
        #exit()
        
        
        #post_life_mask = self.alive(x)

        dx = torch.complex(torch.split(dx, int(dx.shape[3]/2), dim=3)[0], torch.split(dx, int(dx.shape[3]/2), dim=3)[1])
        x = x + dx.transpose(1, 3)
        
        #x = torch.fft.ifft2(x)
        #x = x.transpose(1, 3)


        return x
    
    def update_real(self, x, fire_rate, alive_rate):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """

        dx = x.transpose(1, 3)

        if True: # Diagonal
            alive = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            #alive = (alive + torch.transpose(alive, 2,3)) / 2
            pos_x = alive
            pos_y = torch.transpose(alive, 2,3)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)

            #print(dx.shape)
            dx = torch.concat((dx, pos_x, pos_y, alive_rate), 1)

        

        dx = self.perceive_real(dx)

        dx = dx.transpose(1, 3)

        dx = self.real_fc0_real(dx)

        dx = F.leaky_relu(dx)
        #dx = F.relu(dx)



        dx = self.real_drop0(dx)

        dx = self.real_fc1_real(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate

        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device)) > fire_rate
        stochastic = stochastic.float()
        
        dx = dx * stochastic

        dx = dx.transpose(1, 3)

        x = x + dx.transpose(1, 3)

        return x
    
    def update_dict(self, x, fire_rate, alive_rate, model_dict):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """

        dx = x#.transpose(1, 3)

        if True: # Diagonal
            alive = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            #alive = (alive + torch.transpose(alive, 2,3)) / 2
            pos_x = alive
            pos_y = torch.transpose(alive, 2,3)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)

            #if fourier:
            #    dx = torch.concat((dx.real, dx.imag), 1)

            #print(dx.shape)
            dx = torch.concat((dx, pos_x, pos_y, alive_rate), 1)



        #print(dx.shape)
        dx = model_dict["normalisation"](dx)

        dx = self.perceive_dict(dx, model_dict["conv0"], model_dict["conv1"])

        dx = dx.transpose(1, 3)

        dx = model_dict["fc0"](dx)

        dx = F.leaky_relu(dx)
        #dx = F.relu(dx)



        dx = model_dict["dropout"](dx)

        dx = model_dict["fc1"](dx)

        if fire_rate is None:
            fire_rate = self.fire_rate

        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device)) > fire_rate
        stochastic = stochastic.float()
        
        dx = dx * stochastic

        dx = dx.transpose(1, 3)

        #if fourier:
        #    dx = torch.complex(torch.split(dx, int(dx.shape[3]/2), dim=3)[0], torch.split(dx, int(dx.shape[3]/2), dim=3)[1])

        x = x + dx #.transpose(1, 3)

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
        #print("FFT", torch.max(x), torch.min(x))
        x = torch.fft.fft2(x, norm="forward")#, norm="forward")
        #print("FFT_AFTER", torch.max(x.real), torch.min(x.real), torch.max(x.imag), torch.min(x.imag))
        
        
        #min_real, max_real = -500, 500 #torch.max(x.real), torch.min(x.real)
        #min_imag, max_imag = -256, 256 #torch.max(x.imag), torch.min(x.imag)
        #x.real = (x.real - min_real) / (max_real - min_real)
        #x.imag = (x.imag - min_imag) / (max_imag - min_imag)

        #steps = 45

        x_old = x.clone()
        x = x[..., 0:int(x_old.shape[2]/5), 0:int(x_old.shape[3]/5)]
        #print(x.shape)

        x = torch.concat((x.real, x.imag), 1)
        for step in range(steps):
            #print(x.shape, scaled_t.shape)
            
            #x_update = self.update(x, fire_rate, alive_rate=1-(step/x.shape[2])) 
            x = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_0) 

        x = x.transpose(1, 3)
        x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
        x = x.transpose(1, 3)

            #x = torch.concat((x_update[..., 0:-6], x[..., -6:]), 3)#x_update
            #x = x_update

        x_old[..., 0:int(x_old.shape[2]/5), 0:int(x_old.shape[3]/5)] = x[..., 0:int(x_old.shape[2]/5), 0:int(x_old.shape[3]/5)]
        x = x_old

        #x.real = x.real * (max_real - min_real) + min_real#(x.real - min_real) / max_real
        #x.imag = x.imag * (max_imag - min_imag) + min_imag
        
        x = torch.fft.ifft2(x, norm="forward")#, norm="forward")

        x = x.to(torch.float) #double

        #raise Exception("STOP!")

        for step in range(steps):#int(steps/2)):
            #x_update = self.update_real(x, fire_rate, alive_rate=t) 
            #x = x_update
            x = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_1) 

        x = x.transpose(1, 3)

        #x = x.to(torch.double)


            #x = torch.concat((x_update[..., :3], x[..., 3:4], x_update[..., 4:]), 3) # Leave 3:4
        #if self.complex:
        #    x = torch.fft.ifftn(x)
        #x = x.type(torch.float)
        
        return x
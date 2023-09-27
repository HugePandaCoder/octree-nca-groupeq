import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
 
class DiffusionNCA_fft2(nn.Module):
    r"""Implementation of Diffusion NCA
    """

    class EfficientBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__() 
            self.l0 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l5 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l6 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l7 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)  
            self.l8 = nn.Conv2d(hidden_size*8, hidden_size, kernel_size=1, stride=1, padding=0)   

        def forward(self, input):
            input = input.transpose(1, 3)
            out0 = self.l0(input)
            out1 = self.l1(input)
            out2 = self.l2(input)
            out3 = self.l3(input)
            out4 = self.l4(input)
            out5 = self.l5(input)
            out6 = self.l6(input)
            out7 = self.l7(input)   
            
            out = torch.cat((out0,out1,out2,out3,out4,out5,out6,out7),1)
            out = F.leaky_relu(out)
            return self.l8(out).transpose(1,3) 
        
    class ResNetBlock(nn.Module):
        def __init__(self, hidden_size):
            super().__init__() 

            self.l0 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)
            self.l1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, stride=1, padding=0)

            self.gn0 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
            self.gn1 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)

        def forward(self, input):

            input = input.transpose(1,3)
            out = self.l0(input)
            out = self.gn0(out)
            out = F.leaky_relu(out)
            out = self.l1(out)
            out = out + input
            out = self.gn1(out)
            out = F.leaky_relu(out)

            return out.transpose(1,3)


    #def createModel(self, kernel_size=3, )

    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA_fft2, self).__init__() #  channel_n, fire_rate, device, hidden_size)

        extra_channels = 4

        self.device=device
        self.input_channels = input_channels
        self.channel_n = channel_n

        kernelSize = 3
        padding = int((kernelSize-1)/2)

        # ---------------- MODEL 0 -----------------
        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        self.p0_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="circular")#, groups=channel_n*2+extra_channels*4)#, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        self.p1_real = 0#nn.Conv2d(channel_n*8+extra_channels, channel_n*8, kernel_size=kernelSize, stride=1, padding=padding)#, groups=channel_n*2+extra_channels*4)#, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        self.fc0_real = nn.Conv2d(channel_n*2*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0) #nn.Linear(channel_n*3*2+extra_channels*3, hidden_size)
        #self.fc05_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.fc06_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.fc07_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        self.fc1_real = nn.Conv2d(hidden_size, channel_n*2, kernel_size=1, stride=1, padding=0) #nn.Linear(hidden_size, channel_n*2, bias=False)

        # combine pos and timestep
        #self.effBlock = self.EfficientBlock(hidden_size) 
        self.conv_pt_0 = nn.Sequential(
            nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0)
        )#nn.Conv2d(extra_channels, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_2 = nn.Conv2d(16, extra_channels, kernel_size=1, stride=1, padding=0)

        self.model_0 = {"dropout": self.drop0, "normalisation":self.norm_real2, "conv0": self.p0_real, "conv1": self.p1_real, "fc0": self.fc0_real, "fc1": self.fc1_real, "pt0": self.conv_pt_0}#, "pt1": self.conv_pt_1, "pt2": self.conv_pt_2}#, "fc05": self.fc05_middle_real, "fc06": self.fc05_middle_real, "fc07": self.fc05_middle_real}

        kernelSize = 3
        padding = int((kernelSize-1)/2)

        if False:
            self.fc_mid = nn.Linear(hidden_size, hidden_size)
            self.norm_mid = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
            self.fc_mid2 = nn.Linear(hidden_size, hidden_size)
            self.norm_mid2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        #self.fc_mid3 = nn.Linear(hidden_size, hidden_size)

        self.bn = nn.BatchNorm2d(hidden_size)

        # ---------------- MODEL 1 -----------------
        self.real_drop0 = nn.Dropout(drop_out_rate)
        self.real_norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        self.real_p0_real = nn.Conv2d(channel_n+extra_channels, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="circular")#, groups=channel_n+extra_channels)#reflect, groups=channel_n*2)
        self.real_p1_real = 0#nn.Conv2d(channel_n+extra_channels, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect")#, groups=channel_n+extra_channels)#, groups=channel_n*2)
        self.real_fc0_real = nn.Conv2d(channel_n*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0) #nn.Linear(channel_n*3+extra_channels*3, hidden_size)
        #self.real_fc05_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.real_fc06_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.real_fc07_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        self.real_fc1_real = nn.Conv2d(hidden_size, channel_n, kernel_size=1, stride=1, padding=0) #nn.Linear(hidden_size, channel_n, bias=False)

        # combine pos and timestep
        #self.effBlock_real = self.EfficientBlock(hidden_size)
        self.conv_pt_0_real = nn.Sequential(
            nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0)
        )#= nn.Conv2d(extra_channels, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_1_real = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_2_real = nn.Conv2d(16, extra_channels, kernel_size=1, stride=1, padding=0)

        self.model_1 = {"dropout": self.real_drop0, "normalisation":self.real_norm_real2, "conv0": self.real_p0_real, "conv1": self.real_p1_real, "fc0": self.real_fc0_real, "fc1": self.real_fc1_real, "pt0": self.conv_pt_0_real}#, "pt1": self.conv_pt_1_real, "pt2": self.conv_pt_2_real}#, "fc05": self.real_fc05_middle_real, "fc06": self.real_fc05_middle_real, "fc07": self.real_fc05_middle_real}
        

        # DNA Network
        if True:
            self.dna_lay1 = nn.Conv2d(channel_n, hidden_size*2, kernel_size=5, stride=1, padding=2, padding_mode="circular")
            self.dna_lay2 = nn.GroupNorm(num_groups =  1, num_channels=channel_n)

            self.dna_lay3 = nn.Conv2d(hidden_size*2, hidden_size*2, kernel_size=1, stride=1, padding=0)
            self.dna_lay4 = nn.Conv2d(hidden_size*2, channel_n, kernel_size=5, stride=1, padding=2, padding_mode="circular")#nn.ConvTranspose2d(hidden_size*10, channel_n, kernel_size=5, stride=5, padding=0)

            self.dna_dropout = nn.Dropout(drop_out_rate)

        if False:
            # ---------------- MODEL 2 -----------------
            self.model2_drop0 = nn.Dropout(drop_out_rate)
            self.model2_norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=channel_n*2+extra_channels)
            self.model2_p0_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2+extra_channels, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect", groups=channel_n*2+extra_channels)#reflect, groups=channel_n*2)
            self.model2_p1_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2+extra_channels, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
            self.model2_fc0_real = nn.Linear(channel_n*3*2+extra_channels*3, hidden_size)
            self.model2_fc1_real = nn.Linear(hidden_size, channel_n*2, bias=False)

            self.model_2 = {"dropout": self.model2_drop0, "normalisation":self.model2_norm_real2, "conv0": self.model2_p0_real, "conv1": self.model2_p1_real, "fc0": self.model2_fc0_real, "fc1": self.model2_fc1_real}

            # ---------------- MODEL 3 -----------------
            self.model3_drop0 = nn.Dropout(drop_out_rate)
            self.model3_norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=channel_n+extra_channels)
            self.model3_p0_real = nn.Conv2d(channel_n+extra_channels, channel_n+extra_channels, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect", groups=channel_n+extra_channels)#reflect, groups=channel_n*2)
            self.model3_p1_real = nn.Conv2d(channel_n+extra_channels, channel_n+extra_channels, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect", groups=channel_n+extra_channels)#, groups=channel_n*2)
            self.model3_fc0_real = nn.Linear(channel_n*3+extra_channels*3, hidden_size)
            self.model3_fc1_real = nn.Linear(hidden_size, channel_n, bias=False)

            self.model_3 = {"dropout": self.model3_drop0, "normalisation":self.model3_norm_real2, "conv0": self.model3_p0_real, "conv1": self.model3_p1_real, "fc0": self.model3_fc0_real, "fc1": self.model3_fc1_real}

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
        #y2 = conv1(x)
        y = torch.cat((x,y1),1)#torch.cat((x,y1,y2),1)
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

    def update_dict(self, x, fire_rate, alive_rate, model_dict, step = 0):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """

        dx = x#.transpose(1, 3)

        if True: # Diagonal
            #alive = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            #alive = (alive + torch.transpose(alive, 2,3)) / 2
            
            pos_x = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            #alive = (alive + torch.transpose(alive, 2,3)) / 2
            
            #pos_y = torch.transpose(pos_x, 2,3)
            pos_y = torch.linspace(1, 0, dx.shape[2]).expand(dx.shape[0], 1, dx.shape[3], dx.shape[2]).to(self.device).transpose(2,3)#torch.transpose(alive, 2,3)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
            step = torch.tensor(step).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            #pos_x = alive
            #pos_y = torch.transpose(alive, 2,3)
            #alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
            #step = torch.tensor(step).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)

            #if fourier:
            #    dx = torch.concat((dx.real, dx.imag), 1)

            #print(dx.shape)

            pos_t_enc = self.channel_embedding_4d(torch.concat((pos_x, pos_y, alive_rate, step), 1))
            pos_t_enc = model_dict["pt0"](pos_t_enc)
            #pos_t_enc = F.leaky_relu(pos_t_enc)
            #pos_t_enc = model_dict["pt1"](pos_t_enc)
            #pos_t_enc = F.leaky_relu(pos_t_enc)
            #pos_t_enc = model_dict["pt2"](pos_t_enc)
            #pos_t_enc = F.leaky_relu(pos_t_enc)

            dx = torch.concat((dx, pos_t_enc), 1)



        #print(dx.shape)

        dx = self.perceive_dict(dx, model_dict["conv0"], model_dict["conv1"])

        #dx = dx.transpose(1, 3)
        #dx = F.leaky_relu(dx)
        #dx = dx.transpose(1, 3)

        
        

        dx = model_dict["fc0"](dx)


        #dx = self.bn(dx)


        dx = model_dict["normalisation"](dx)
        dx = dx.transpose(1, 3)
        dx = F.leaky_relu(dx)

        #dx = F.relu(dx)



        #dx = model_dict["dropout"](dx)

        #dx = model_dict["effBlock"](dx)
        #dx = F.leaky_relu(dx)

        #dx = model_dict["fc05"](dx)
        #dx = F.leaky_relu(dx)
        #dx = model_dict["fc06"](dx)
        #dx = F.leaky_relu(dx)
        #dx = model_dict["fc07"](dx)
        #dx = F.leaky_relu(dx)
        #dx = model_dict["dropout"](dx)

        dx = dx.transpose(1, 3)
        dx = model_dict["fc1"](dx)
        dx = dx.transpose(1, 3)

        #dx = F.leaky_relu(dx)

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
    
    def forward(self, x, steps=10, fire_rate=None, t=0, epoch=0, **kwargs):
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


        # ---------------- MODEL 0 -----------------
        if False:
            x = x.transpose(1, 3) 
            x = torch.fft.fft2(x, norm="forward")#, norm="forward")

            x_old = x.clone()
            factor = 5
            pixel_X = 20#int(x_old.shape[2]/factor)
            pixel_Y = 20#int(x_old.shape[3]/factor)
            x = x[..., 0:pixel_X, 0:pixel_Y]

            x = torch.concat((x.real, x.imag), 1)
            for step in range(steps):
                x = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_0) 

            x = x.transpose(1, 3)
            x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
            x = x.transpose(1, 3)

            x_old[..., 0:pixel_X, 0:pixel_Y] = x[..., 0:pixel_X, 0:pixel_Y]
            x = x_old
            x = torch.fft.ifft2(x, norm="forward").real #.to(torch.float)#, norm="forward")
            #x = x.to(torch.float) #double

            
            # ---------------- MODEL 1 -----------------
            for step in range(steps):#int(steps/2)):
                #x_update = self.update_real(x, fire_rate, alive_rate=t) 
                #x = x_update
                x = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_1) 
            x = x.transpose(1, 3)
        else:
            x = x.transpose(1, 3) 
            
            #fire_rate = 0
            
            factor = 5
            pixel_X = 16#int(x_old.shape[2]/factor)
            pixel_Y = 16#int(x_old.shape[3]/factor)
            x = torch.fft.fft2(x, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
            x = torch.fft.fftshift(x, dim=(2,3))
            x_old = x.clone()
            x_start, y_start = x.shape[2]//2, x.shape[3]//2 # - pixel_X//2 - pixel_Y//2, 
            x = x[..., x_start:x_start+pixel_X, y_start:y_start+pixel_Y]


            if False:
                # FLIP X
                x = torch.concat(
                    (torch.split(x, int(x.shape[2]//2), dim=2)[0], 
                    torch.flip(torch.split(x, int(x.shape[2]//2), dim=2)[1], dims=[2])
                    ), 1)
                # FLIP Y
                x = torch.concat(
                    (torch.split(x, int(x.shape[3]//2), dim=3)[0], 
                    torch.flip(torch.split(x, int(x.shape[3]//2), dim=3)[1], dims=[3])
                    ), 1)

            steps_f = pixel_X
            x = torch.concat((x.real, x.imag), 1)
            for step in range(int(steps_f*1.5)):
                x_new = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_0, step=step/(steps_f)) 
                x[:, self.input_channels:self.channel_n, ...] = x_new[:, self.input_channels:self.channel_n, ...]
                x[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...] = x_new[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...]

            if False:
                # BACK FLIP X
                x = torch.concat(
                    (torch.split(x, int(x.shape[1]//2), dim=1)[0], 
                    torch.flip(
                    torch.split(x, int(x.shape[1]//2), dim=1)[1], dims=[3])
                    ), 3)
                # BACK FLIP Y
                x = torch.concat(
                    (torch.split(x, int(x.shape[1]//2), dim=1)[0], 
                    torch.flip(
                    torch.split(x, int(x.shape[1]//2), dim=1)[1], dims=[2])
                    ), 2)


            x = x.transpose(1, 3)
            x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
            x = x.transpose(1, 3)

            x_old[:, self.input_channels:, x_start:x_start+pixel_X, y_start:y_start+pixel_Y] = x[:, self.input_channels:, ...]
            #x = x_old
            x_old = torch.fft.ifftshift(x_old, dim=(2,3))
            x = torch.fft.ifft2(x_old, norm="forward").real #.to(torch.float)#, norm="forward") #, norm="forward"
            #x[:, 0:3, ...] = x_old[:, 0:3, ...]
            #x = x.to(torch.float) #double

            
            # ---------------- MODEL 1 -----------------
           # steps = min(1 + epoch // 45, steps)
            for step in range(steps):#int(steps/2)):
                #x_update = self.update_real(x, fire_rate, alive_rate=t) 
                #x = x_update
                x[:, self.input_channels:, ...] = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_1, step=step/steps)[:, self.input_channels:, ...] 
            
            x_up = self.dna_lay2(x)
            x_up[:, 0:self.input_channels, ...] = x[:, 0:self.input_channels, ...]
            #print(x_up.shape)
            x_up = self.dna_lay1(x_up)
            
            #print(x_up.shape)
            x_up = F.leaky_relu(x_up)
            x_up[:, 0:self.input_channels, ...] = x[:, 0:self.input_channels, ...]
            #x_up = self.dna_dropout(x_up)
            x_up = self.dna_lay3(x_up)
            
            #print(x_up.shape)
            x_up = F.leaky_relu(x_up)
            x_up[:, 0:self.input_channels, ...] = x[:, 0:self.input_channels, ...]
            x = self.dna_lay4(x_up)
            
            x = x.transpose(1, 3)

        if False:
            # ---------------- MODEL 2 -----------------
            #x = x.transpose(1, 3) 
            x = torch.fft.fft2(x, norm="forward")#, norm="forward")

            x_old = x.clone()
            x = x[..., 0:int(x_old.shape[2]/5), 0:int(x_old.shape[3]/5)]

            x = torch.concat((x.real, x.imag), 1)
            for step in range(steps):
                x = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_2) 

            x = x.transpose(1, 3)
            x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
            x = x.transpose(1, 3)

            x_old[..., 0:int(x_old.shape[2]/5), 0:int(x_old.shape[3]/5)] = x[..., 0:int(x_old.shape[2]/5), 0:int(x_old.shape[3]/5)]
            x = x_old
            x = torch.fft.ifft2(x, norm="forward")#, norm="forward")
            x = x.to(torch.float) #double

            # ---------------- MODEL 3 -----------------
            for step in range(steps):#int(steps/2)):
                #x_update = self.update_real(x, fire_rate, alive_rate=t) 
                #x = x_update
                x = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_3) 

        #x = x.to(torch.double)


            #x = torch.concat((x_update[..., :3], x[..., 3:4], x_update[..., 4:]), 3) # Leave 3:4
        #if self.complex:
        #    x = torch.fft.ifftn(x)
        #x = x.type(torch.float)
        
        return x
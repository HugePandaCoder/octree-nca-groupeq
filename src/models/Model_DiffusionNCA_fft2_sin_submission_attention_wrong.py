import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
 
class ScaledDotProductAttentionModule(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(ScaledDotProductAttentionModule, self).__init__()
        # Transformations for query, key, and value
        self.query_transform = nn.Linear(input_dim, attention_dim)
        self.key_transform = nn.Linear(input_dim, attention_dim)
        self.value_transform = nn.Linear(input_dim, attention_dim)
        self.attention_dim = attention_dim

    def forward(self, x):
        # Transform inputs to query, key, value
        Q = self.query_transform(x)  # Query
        K = self.key_transform(x)    # Key
        V = self.value_transform(x)  # Value

        # Compute scaled dot-product attention
        # Scaled by dividing by the square root of the dimension of the key
        scale = torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the attention weights to the values
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

class DiffusionNCA_fft2_attention(nn.Module):
    r"""Implementation of Diffusion NCA
    """


    #def createModel(self, kernel_size=3, )

    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA_fft2_attention, self).__init__() #  channel_n, fire_rate, device, hidden_size)

        extra_channels = 4

        self.device=device
        self.input_channels = input_channels
        self.channel_n = channel_n

        kernelSize = 3
        padding = int((kernelSize-1)/2)

        # ---------------- MODEL 0 -----------------
        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        self.p0_real = nn.Conv2d(channel_n*2, channel_n*2, kernel_size=kernelSize, stride=1, padding=padding)#, groups=channel_n*2+extra_channels*4)#, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        self.p1_real = 0#nn.Conv2d(channel_n*8+extra_channels, channel_n*8, kernel_size=kernelSize, stride=1, padding=padding)#, groups=channel_n*2+extra_channels*4)#, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        self.fc0_real = nn.Conv2d(channel_n*2*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0) #nn.Linear(channel_n*3*2+extra_channels*3, hidden_size)
        #self.fc05_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.fc06_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.fc07_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        self.fc1_real = nn.Conv2d(hidden_size, channel_n*2, kernel_size=1, stride=1, padding=0) #nn.Linear(hidden_size, channel_n*2, bias=False)

        self.attention = ScaledDotProductAttentionModule(channel_n*2*2+extra_channels, channel_n*2*2+extra_channels)

        # combine pos and timestep
        #self.effBlock = self.EfficientBlock(hidden_size) 
        self.conv_pt_0 = nn.Sequential(
            nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0)
        )#nn.Conv2d(extra_channels, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_2 = nn.Conv2d(16, extra_channels, kernel_size=1, stride=1, padding=0)

        self.model_0 = {"dropout": self.drop0, "normalisation":self.norm_real2, "conv0": self.p0_real, "conv1": self.p1_real, "fc0": self.fc0_real, "fc1": self.fc1_real, "pt0": self.conv_pt_0, "attention": self.attention}#, "pt1": self.conv_pt_1, "pt2": self.conv_pt_2}#, "fc05": self.fc05_middle_real, "fc06": self.fc05_middle_real, "fc07": self.fc05_middle_real}

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
        self.real_p0_real = nn.Conv2d(channel_n, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect")#, groups=channel_n+extra_channels)#reflect, groups=channel_n*2)
        self.real_p1_real = 0#nn.Conv2d(channel_n+extra_channels, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect")#, groups=channel_n+extra_channels)#, groups=channel_n*2)
        self.real_fc0_real = nn.Conv2d(channel_n*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0) #nn.Linear(channel_n*3+extra_channels*3, hidden_size)
        #self.real_fc05_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.real_fc06_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.real_fc07_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        self.real_fc1_real = nn.Conv2d(hidden_size, channel_n, kernel_size=1, stride=1, padding=0) #nn.Linear(hidden_size, channel_n, bias=False)

        self.attention_real = ScaledDotProductAttentionModule(channel_n*2+extra_channels, channel_n*2+extra_channels)

        # combine pos and timestep
        #self.effBlock_real = self.EfficientBlock(hidden_size)
        self.conv_pt_0_real = nn.Sequential(
            nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0)
        )#= nn.Conv2d(extra_channels, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_1_real = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_2_real = nn.Conv2d(16, extra_channels, kernel_size=1, stride=1, padding=0)

        self.model_1 = {"dropout": self.real_drop0, "normalisation":self.real_norm_real2, "conv0": self.real_p0_real, "conv1": self.real_p1_real, "fc0": self.real_fc0_real, "fc1": self.real_fc1_real, "pt0": self.conv_pt_0_real, "attention": self.attention_real}#, "pt1": self.conv_pt_1_real, "pt2": self.conv_pt_2_real}#, "fc05": self.real_fc05_middle_real, "fc06": self.real_fc05_middle_real, "fc07": self.real_fc05_middle_real}
        
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

    def perceive_dict(self, x, conv0, conv1):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = conv0(x.clone())
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
        #dx = torch.concat((dx, dx), 1)
        dx = self.perceive_dict(x, model_dict["conv0"], model_dict["conv1"])


        if True: # Diagonal
            pos_x = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            pos_y = torch.linspace(1, 0, dx.shape[2]).expand(dx.shape[0], 1, dx.shape[3], dx.shape[2]).to(self.device).transpose(2,3)#torch.transpose(alive, 2,3)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
            step = torch.tensor(step).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            pos_t_enc = torch.concat((pos_x, pos_y, alive_rate, step), 1) #self.channel_embedding_4d(torch.concat((pos_x, pos_y, alive_rate, step), 1))
            #pos_t_enc = model_dict["pt0"](pos_t_enc)

            dx = torch.concat((dx, pos_t_enc), 1)
        
        # Attention here

        if False:      
            #dx = dx.view(dx.size(0), dx.size(2) * dx.size(3), dx.size(1))
            dx = dx.transpose(1,3)
            dx = dx + model_dict["attention"](dx)
            #dx = dx.view(dx.size(0), -1, 16, 16)
            dx = dx.transpose(1,3)


        dx = model_dict["fc0"](dx)

        dx = model_dict["normalisation"](dx)
        dx = dx.transpose(1, 3)
        dx = F.leaky_relu(dx)

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

        x = x.transpose(1, 3).to(self.device) 
        
        #fire_rate = 0
        
        if True: # NO FOURIER
            factor = 5
            pixel_X = 16#int(x_old.shape[2]/factor)
            pixel_Y = 16#int(x_old.shape[3]/factor)
            x = torch.fft.fft2(x, norm="forward")#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
            x = torch.fft.fftshift(x, dim=(2,3))
            x_old = x.clone()
            x_start, y_start = x.shape[2]//2, x.shape[3]//2 # - pixel_X//2 - pixel_Y//2, 
            x = x[..., x_start:x_start+pixel_X, y_start:y_start+pixel_Y]

            steps_f = pixel_X
            x = torch.concat((x.real, x.imag), 1)
            #for step in range(steps_f):
            #    x_new = self.update_dict(x, 0, alive_rate=t, model_dict=self.model_0, step=step/(steps_f)) 
            #    x[:, self.input_channels:self.channel_n, ...] = x_new[:, self.input_channels:self.channel_n, ...]
            #    x[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...] = x_new[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...]

            x = x.transpose(1, 3)
            x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
            x = x.transpose(1, 3)

            x_old[:, self.input_channels:, x_start:x_start+pixel_X, y_start:y_start+pixel_Y] = x[:, self.input_channels:, ...]
            x_old = torch.fft.ifftshift(x_old, dim=(2,3))
            x = torch.fft.ifft2(x_old, norm="forward").real 


                
            # ---------------- MODEL 1 -----------------
            for step in range(steps):#int(steps/2)):
                x[:, self.input_channels:, ...] = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.model_1, step=step/steps)[:, self.input_channels:, ...] 
            x = x.transpose(1, 3)
        
        return x

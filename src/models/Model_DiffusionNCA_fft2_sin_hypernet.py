import torch
import torch.nn as nn
from src.models.Model_BackboneNCA import BackboneNCA
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
import torch.nn.utils.weight_norm as weight_norm

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

class AttentionModule(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionModule, self).__init__()
        self.attention_fc = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        attention_weights = F.softmax(self.attention_fc(x), dim=1)
        return attention_weights

class HyperNetwork2(nn.Module):
    def __init__(self, input_size, channel_n, kernel_size, hidden_size, extra_channels):#(self, input_dim, hidden_dim, attention_dim, output_dims):
        super(HyperNetwork2, self).__init__()

        attention_dim = 8
        hidden_size_hyper = 128

        fourier_fac = 2
        self.conv3d_fourier = (channel_n*fourier_fac+extra_channels) * kernel_size * kernel_size 
        self.fc0_fourier = (channel_n*2*fourier_fac+extra_channels*2) * hidden_size 
        self.fc1_fourier = (hidden_size) * channel_n *fourier_fac

        self.conv3d_image = (channel_n+extra_channels) * kernel_size * kernel_size
        self.fc0_image = (channel_n*2+extra_channels*2) * hidden_size
        self.fc1_image = (hidden_size) * channel_n


        self.attention = AttentionModule(input_size, attention_dim)
        self.fc1 = weight_norm(nn.Linear(input_size + attention_dim, hidden_size_hyper))
        self.fc2 = weight_norm(nn.Linear(hidden_size_hyper, hidden_size_hyper))  # Adjusted to sum of output dims
        self.dropout = nn.Dropout(0.25)
        self.batch_norm = nn.BatchNorm1d(hidden_size_hyper)

        # Additional layers for weight generation
        self.lin05_conv3d_fourier = weight_norm(nn.Linear(hidden_size_hyper, self.conv3d_fourier))
        self.lin05_fc0_fourier = weight_norm(nn.Linear(hidden_size_hyper, self.fc0_fourier))
        self.lin05_fc1_fourier = weight_norm(nn.Linear(hidden_size_hyper, self.fc1_fourier))

        self.lin05_conv3d_image = weight_norm(nn.Linear(hidden_size_hyper, self.conv3d_image))
        self.lin05_fc0_image = weight_norm(nn.Linear(hidden_size_hyper, self.fc0_image))
        self.lin05_fc1_image = weight_norm(nn.Linear(hidden_size_hyper, self.fc1_image))

    def forward(self, x):
        attention_weights = self.attention(x)
        x = torch.cat([x, attention_weights], dim=1)
        x = F.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Output of fc2 not directly used

        # Generate weights for each specified layer
        generated_weights = {'fourier': {}, 'image': {}}
        generated_weights["fourier"]['conv3d'] = self.lin05_conv3d_fourier(x)
        generated_weights["fourier"]['fc0'] = self.lin05_fc0_fourier(x)
        generated_weights["fourier"]['fc1'] = self.lin05_fc1_fourier(x)

        generated_weights["image"]['conv3d'] = self.lin05_conv3d_image(x)
        generated_weights["image"]['fc0'] = self.lin05_fc0_image(x)
        generated_weights["image"]['fc1'] = self.lin05_fc1_image(x)

        return generated_weights


class HyperNetwork(nn.Module):
    def __init__(self, input_size, channel_n, kernel_size, hidden_size, extra_channels, attention_dim=32):#(self, input_dim, hidden_dim, attention_dim, output_dims):
        super(HyperNetwork, self).__init__()
        fourier_fac = 2
        self.conv3d_fourier = (channel_n*fourier_fac+extra_channels) * kernel_size * kernel_size 
        self.fc0_fourier = (channel_n*2*fourier_fac+extra_channels*2+attention_dim) * hidden_size 
        self.fc1_fourier = (hidden_size) * channel_n *fourier_fac
        
        #self.fc2_fourier = (hidden_size) * channel_n *fourier_fac
        #self.fc3_fourier = (hidden_size) * channel_n *fourier_fac

        self.conv3d_image = (channel_n+extra_channels) * kernel_size * kernel_size
        self.fc0_image = (channel_n*2+extra_channels*2+attention_dim) * hidden_size 
        self.fc1_image = (hidden_size) * channel_n

        #self.fc2_image = (hidden_size) * channel_n
        #self.fc3_image = (hidden_size) * channel_n

        #output_size =  self.conv3d + self.fc0 + self.fc1
            
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


        self.lin01 = weight_norm(nn.Linear(input_size, 128))
        #input_size = 0
        self.lin02 = weight_norm(nn.Linear(128+input_size, 128))
        #self.lin03 = nn.Linear(64+input_size, 256)
        #self.lin04 = nn.Linear(256+input_size, 4096)
        self.lin05_conv3d_fourier = weight_norm(nn.Linear(128+input_size, self.conv3d_fourier))
        self.lin05_fc0_fourier = weight_norm(nn.Linear(128+input_size, self.fc0_fourier))
        self.lin05_fc1_fourier = weight_norm(nn.Linear(128+input_size, self.fc1_fourier))

        #self.lin05_fc2_fourier = nn.Linear(128+input_size, self.fc2_fourier)
        #self.lin05_fc3_fourier = nn.Linear(128+input_size, self.fc3_fourier)

        self.lin05_conv3d_image = weight_norm(nn.Linear(128+input_size, self.conv3d_image))
        self.lin05_fc0_image = weight_norm(nn.Linear(128+input_size, self.fc0_image))
        self.lin05_fc1_image = weight_norm(nn.Linear(128+input_size, self.fc1_image))

        #self.lin05_fc2_image = nn.Linear(128+input_size, self.fc2_image)
        #self.lin05_fc3_image = nn.Linear(128+input_size, self.fc3_image)

        self.silu = nn.ReLU()
        
    def forward(self, x):
        generated_weights = {'fourier':{}, 'image':{}}

        # Construct weights

        # Generate flattened weights
        dx = self.silu(self.lin01(x))

        dx = torch.cat((dx, x), dim=1)
        dx = self.silu(self.lin02(dx))
        dx = torch.cat((dx, x), dim=1)

        generated_weights["fourier"]['conv3d'] = self.lin05_conv3d_fourier(dx)
        generated_weights["fourier"]['fc0'] = self.lin05_fc0_fourier(dx)
        generated_weights["fourier"]['fc1'] = self.lin05_fc1_fourier(dx)

        #generated_weights["fourier"]['fc2'] = self.lin05_fc2_fourier(dx)
        #generated_weights["fourier"]['fc3'] = self.lin05_fc3_fourier(dx)

        generated_weights["image"]['conv3d'] = self.lin05_conv3d_image(dx)
        generated_weights["image"]['fc0'] = self.lin05_fc0_image(dx)
        generated_weights["image"]['fc1'] = self.lin05_fc1_image(dx)

        #generated_weights["image"]['fc2'] = self.lin05_fc2_image(dx)
        #generated_weights["image"]['fc3'] = self.lin05_fc3_image(dx)

        return generated_weights

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Calculate the square root of the sum of the squares of each pixel
        # Add epsilon for numerical stability
        norm = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
        # Normalize the input x
        x = x / norm
        return x


class DiffusionNCA_fft2_hypernet(nn.Module):
    r"""Implementation of Diffusion NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, drop_out_rate=0.25, img_size=28):
        r"""Init function
        """
        super(DiffusionNCA_fft2_hypernet, self).__init__() #  channel_n, fire_rate, device, hidden_size)

        extra_channels = 4
        self.extra_channels = extra_channels
        self.attention = False


        if self.attention:
            self.attention_dim = 32
        else:
            self.attention_dim = 0

        self.device=device
        self.input_channels = input_channels
        self.channel_n = channel_n
        self.hidden_size = hidden_size

        self.kernel_size = kernelSize = 7
        self.padding = padding = int((kernelSize-1)/2)
        

        if self.attention:
            self.attention = ScaledDotProductAttentionModule(channel_n*2*2+2*extra_channels, self.attention_dim) #channel_n*2*2+2*extra_channels)
            self.attention_real = ScaledDotProductAttentionModule(channel_n*2+2*extra_channels, self.attention_dim) #channel_n*2+2*extra_channels)

        self.generated_weights = {}
        self.hypernetwork = HyperNetwork(1, channel_n, kernelSize, hidden_size, extra_channels, attention_dim=self.attention_dim) #41

        # ---------------- MODEL 0 -----------------
        self.drop0 = nn.Dropout(drop_out_rate)
        self.norm_real2 = PixelNorm()#nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        
        self.p0_real = nn.Conv2d(channel_n*2+extra_channels, channel_n*2, kernel_size=kernelSize, stride=1, padding=padding)#, groups=channel_n*2+extra_channels*4)#, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        self.p1_real = 0#nn.Conv2d(channel_n*8+extra_channels, channel_n*8, kernel_size=kernelSize, stride=1, padding=padding)#, groups=channel_n*2+extra_channels*4)#, padding_mode="reflect", groups=channel_n*2+extra_channels)#, groups=channel_n*2)
        self.fc0_real = nn.Conv2d(channel_n*2*2+extra_channels, hidden_size*16, kernel_size=1, stride=1, padding=0) #nn.Linear(channel_n*3*2+extra_channels*3, hidden_size)
        #self.fc05_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.fc06_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.fc07_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        self.fc1_real = nn.Conv2d(hidden_size*16, channel_n*2, kernel_size=1, stride=1, padding=0) #nn.Linear(hidden_size, channel_n*2, bias=False)

        # combine pos and timestep
        #self.effBlock = self.EfficientBlock(hidden_size) 
        self.conv_pt_0 = nn.Sequential(
            nn.Conv2d(extra_channels*4, 32, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(32, extra_channels, kernel_size=1, stride=1, padding=0)
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
        #self.real_norm_real2 = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        self.real_norm_real2 = PixelNorm()
        self.real_p0_real = nn.Conv2d(channel_n+extra_channels, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect")#, groups=channel_n+extra_channels)#reflect, groups=channel_n*2)
        self.real_p1_real = 0#nn.Conv2d(channel_n+extra_channels, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect")#, groups=channel_n+extra_channels)#, groups=channel_n*2)
        self.real_fc0_real = nn.Conv2d(channel_n*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0) #nn.Linear(channel_n*3+extra_channels*3, hidden_size)
        #self.real_fc05_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.real_fc06_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        #self.real_fc07_middle_real = self.ResNetBlock(hidden_size)#nn.Linear(hidden_size, hidden_size)
        self.real_fc1_real = nn.Conv2d(hidden_size, channel_n, kernel_size=1, stride=1, padding=0) #nn.Linear(hidden_size, channel_n, bias=False)

        # combine pos and timestep
        #self.effBlock_real = self.EfficientBlock(hidden_size)
        self.conv_pt_0_real = nn.Sequential(
            nn.Conv2d(extra_channels*4, 32, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(32, extra_channels, kernel_size=1, stride=1, padding=0)
        )#= nn.Conv2d(extra_channels, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_1_real = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        #self.conv_pt_2_real = nn.Conv2d(16, extra_channels, kernel_size=1, stride=1, padding=0)

        self.model_1 = {"dropout": self.real_drop0, "normalisation":self.real_norm_real2, "conv0": self.real_p0_real, "conv1": self.real_p1_real, "fc0": self.real_fc0_real, "fc1": self.real_fc1_real, "pt0": self.conv_pt_0_real}#, "pt1": self.conv_pt_1_real, "pt2": self.conv_pt_2_real}#, "fc05": self.real_fc05_middle_real, "fc06": self.real_fc05_middle_real, "fc07": self.real_fc05_middle_real}
        

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

    def perceive_dict(self, x, weights_conv0, fourier_fac=1):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        #y1 = generated_weights(x)
        batch_size = weights_conv0.shape[0]
        output = []
        for i in range(batch_size):
            weights = weights_conv0[i].view(self.channel_n*fourier_fac+self.extra_channels, 1, self.kernel_size, self.kernel_size)
            output.append(F.conv2d(x[i:i+1], weights, padding=self.padding, groups=self.channel_n*fourier_fac+self.extra_channels))
        y1 = torch.cat(output, dim=0)
        y = torch.cat((x,y1),1)
        return y

    def update_dict(self, x, fire_rate, alive_rate, model_dict, step = 0, fourier_fac=1):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """
        dx = x#.transpose(1, 3)

        if True: # Diagonal
            pos_x = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            pos_y = torch.linspace(1, 0, dx.shape[2]).expand(dx.shape[0], 1, dx.shape[3], dx.shape[2]).to(self.device).transpose(2,3)#torch.transpose(alive, 2,3)
            alive_rate = alive_rate.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
            step = torch.tensor(step).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
            pos_t_enc = self.channel_embedding_4d(torch.concat((pos_x, pos_y, alive_rate, step), 1))
            pos_t_enc = model_dict["pt0"](pos_t_enc)
            dx = torch.concat((dx, pos_t_enc), 1)

        dx = self.perceive_dict(dx, model_dict["conv3d"], fourier_fac)

        # Attention here

        if self.attention:      
            #dx = dx.view(dx.size(0), dx.size(2) * dx.size(3), dx.size(1))
            dx = dx.transpose(1,3)
            att = model_dict["attention"](dx)
            #dx = dx.view(dx.size(0), -1, 16, 16)
            att = att.transpose(1,3)
            dx = dx.transpose(1,3)
            dx = torch.concat((dx, att), 1)


        batch_size = dx.shape[0]
        output = []
        for i in range(batch_size):
            weights = model_dict['fc0'][i].view(self.hidden_size, self.channel_n*2*fourier_fac+self.extra_channels*2+self.attention_dim, 1, 1)
            output.append(F.conv2d(dx[i:i+1], weights, padding=0))
        dx = torch.cat(output, dim=0)

        dx = model_dict["normalisation"](dx)
        dx = dx.transpose(1, 3)
        dx = F.leaky_relu(dx)

        dx = dx.transpose(1, 3)
        #dx = model_dict["fc1"](dx)
        output = []
        for i in range(batch_size):
            weights = model_dict['fc1'][i].view(self.channel_n *fourier_fac, self.hidden_size, 1, 1)
            output.append(F.conv2d(dx[i:i+1], weights, padding=0))
        dx = torch.cat(output, dim=0)
        dx = dx.transpose(1, 3)
        skip = dx

        ## ------------------ Add deeper architecture ------------------
        if False:
            dx = dx.transpose(1, 3)
            output = []
            for i in range(batch_size):
                weights = model_dict['fc2'][i].view(self.hidden_size, self.channel_n *fourier_fac, 1, 1)
                output.append(F.conv2d(dx[i:i+1], weights, padding=0))
            dx = torch.cat(output, dim=0)
            
            dx = F.leaky_relu(dx)

            output = []
            for i in range(batch_size):
                weights = model_dict['fc3'][i].view(self.channel_n *fourier_fac, self.hidden_size, 1, 1)
                output.append(F.conv2d(dx[i:i+1], weights, padding=0))
            dx = torch.cat(output, dim=0)
            dx = dx.transpose(1, 3)

        #dx = F.leaky_relu(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate

        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device)) > fire_rate
        stochastic = stochastic.float()
        
        dx = (dx) * stochastic

        dx = dx.transpose(1, 3)

        x = x + dx 

        return x
    
    def forward(self, x, steps=10, fire_rate=None, t=0, epoch=0, properties=None, **kwargs):
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
        t_t = t[..., None]
        if len(t_t.shape) == 1:
            t_t = t_t[..., None]
            properties = properties[None, ...]

        #t_t = torch.cat((t_t, torch.tensor(properties).to(self.device)), 1)
        self.generated_weights = self.hypernetwork(t_t)

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



              
            x = torch.fft.fft2(x, norm="forward")#, s=(pixel_X, pixel_Y))#) #, norm="forward" , s=(x_old.shape[2], x_old.shape[3])
            x_old = x.clone()  
            x = torch.fft.fftshift(x, dim=(2,3))
            
            #x_old = x.clone()
            x_start, y_start = (x.shape[2])//2, (x.shape[3])//2#(x.shape[2]-pixel_X)//2, (x.shape[3]-pixel_Y)//2 # - pixel_X//2 - pixel_Y//2, 
            #x_start, y_start = x.shape[2]//2, x.shape[3]//2
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
            self.generated_weights['fourier']['normalisation'] = self.norm_real2
            self.generated_weights['image']['normalisation'] = self.real_norm_real2
            self.generated_weights['fourier']['pt0'] = self.conv_pt_0
            self.generated_weights['image']['pt0'] = self.conv_pt_0_real

            if self.attention:
                self.generated_weights['fourier']['attention'] = self.attention
                self.generated_weights['image']['attention'] = self.attention_real

            for step in range(steps_f):
                x_new = self.update_dict(x, 0, alive_rate=t, model_dict=self.generated_weights['fourier'], step=step/(steps_f), fourier_fac=2) 
                x[:, self.input_channels:self.channel_n, ...] = x_new[:, self.input_channels:self.channel_n, ...]
                x[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...] = x_new[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...]



            x = x.transpose(1, 3)
            x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
            x = x.transpose(1, 3)

            x = torch.fft.ifftshift(x, dim=(2,3))
            x_old[:, self.input_channels:, x_start:x_start+pixel_X, y_start:y_start+pixel_Y] = x[:, self.input_channels:, ...]
            x = torch.fft.ifft2(x_old, norm="forward").real #, s=(x_old.shape[2], x_old.shape[3])).real 

            
            
            

            #x[:, 0:self.input_channels, ...] = x_old[:, 0:self.input_channels, ...]

            x_four = x
            
            # ---------------- MODEL 1 -----------------
            for step in range(steps):#int(steps/2)):
                x[:, self.input_channels:, ...] = self.update_dict(x, fire_rate, alive_rate=t, model_dict=self.generated_weights['image'], step=step/steps)[:, self.input_channels:, ...] 
            #x = x.transpose(1, 3)
        
        return x.transpose(1, 3), x_four.transpose(1, 3)

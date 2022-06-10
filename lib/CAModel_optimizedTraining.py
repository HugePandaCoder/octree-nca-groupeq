import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CAModel_optimizedTraining(nn.Module):
    def __init__(self, channel_n, fire_rate, device, checkCellsAlive=False, hidden_size=128):
        super(CAModel_optimizedTraining, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.checkCellsAlive = checkCellsAlive
        self.hidden_size = hidden_size

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        torch.nn.init.xavier_uniform(self.fc0.weight)
        torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate

        self.die_after_choice = False
        self.to(self.device)

    def increaseHiddenLayer(self, increase=1, optimizer = None):
        print("____________________________")
        print(self.fc0.bias.shape)
        print(self.fc0.weight.data.shape) 
        print(self.fc1.weight.data.shape) 
        #print(self.fc1.bias.shape)
        #x = torch.zeros(increase, self.channel_n*3)

        # Scale hidden_size by given size
        increase = self.hidden_size
        fc0_copy = self.fc0
        fc1_copy = self.fc1
        self.fc0 = nn.Linear(self.channel_n*3, self.hidden_size+increase)
        self.fc1 = nn.Linear(self.hidden_size+increase, self.channel_n, bias=False)
        self.hidden_size = self.hidden_size+increase

        self.fc0.weight.data = torch.cat((fc0_copy.weight.data, 0.01*torch.nn.init.xavier_uniform(torch.full((increase, fc0_copy.weight.data.size(1)), 0.0001).to(self.device))), dim=0) #.resize_(self.fc0.weight.data.size(0), self.fc0.weight.data.size(1) +1)
        self.fc0.weight.data.requires_grad_(True)
        self.fc0.bias = torch.nn.parameter.Parameter(data=torch.cat((fc0_copy.bias, torch.full((increase,), 0).to(self.device)), dim=0))
        self.fc0.bias.requires_grad_(True)
        self.fc1.weight.data = torch.cat((fc1_copy.weight.data, 0.01*torch.nn.init.xavier_uniform(torch.full((fc1_copy.weight.data.size(0), increase), 0.0001).to(self.device))), dim=1)
        self.fc1.weight.data.requires_grad_(True)
        print(self.fc0.bias.shape)
        print(self.fc0.weight.data.shape)
        print(self.fc1.weight.data.shape)
        #optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        return self

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def choiceMade(self, x):
        return x > 0.5 # F.max_pool2d(x[:, 3, :, :], kernel_size=3, stride=1, padding=1)

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x_in, fire_rate, angle):
        x = x_in.transpose(1,3)

        if self.checkCellsAlive:
            pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        #if self.checkCellsAlive:
        #    #pre_life_mask = self.alive(x)
        #    post_life_mask = self.alive(x)
        #    life_mask = (pre_life_mask & post_life_mask).float()
        #    x = x * life_mask

        x = x.transpose(1,3)

        #if self.die_after_choice:
        #    dead_mask = x_in[..., 3] > 0 #x[..., 3]
        #    x[..., 3][dead_mask] = 1 #x_in[..., 3][dead_mask]
            #x[..., 3][dead_mask] = 1

        return x #.transpose(1,3)

    def forward_old(self, x, steps=1, fire_rate=None, angle=0.0):
        x_sum = x[...,3:6]
        for step in range(steps):
            x_temp = self.update(x, fire_rate, angle)
            x_sum = x_sum + torch.abs(x_temp[..., 3:6] - x[..., 3:6])
            x[...,3:] = x_temp[...,3:]
        return x, x_sum

    def forward_full(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x[...,3:] = self.update(x, fire_rate, angle)[...,3:]
        return x

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x[...,3:] = self.update(x, fire_rate, angle)[...,3:].detach()
        return x

    def forward_opt(self, x, xs, ys, patch_scale, steps=1, fire_rate=None, angle=0.0):
        #self.part_mask = None
        #self.model_no_grad = self.model.__deepcopy__()
        #x2 = x.__deepcopy__()
        #patch_scale = 24
        #xs = torch.randint(0, 64 - patch_scale, (1,))[0] #np.random.randint(0, 64 - patch_scale) #
        #ys = torch.randint(0, 64 - patch_scale, (1,))[0] #np.random.randint(0, 64 - patch_scale) #
        #xs = 0
        #ys = 0
        #xs = 23
        #ys = 23
        
        #xs = 63
        #print(xs)

        #a = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        #b = torch.tensor([[1, 1],[1,1]])

        #print(x.shape)
        #x2 = x.detach().clone()
        print("forward")
        x_part = x[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :].clone()
        x2  = x.clone()
        for step in range(steps):
           

            x2[...,3:] = self.update(x2, fire_rate, angle)[...,3:]
            x2 = x2.detach()

            x_part[...,3:] = self.update(x_part, fire_rate, angle)[...,3:]
            #print(x_part.shape)
            x2_part = x2[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :].clone()
            x2_part[:, 1:-1, 1:-1 :] = x_part[:, 1:-1, 1:-1 :]
            x_part = x2_part
            x2[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :] = x_part.detach()
            
            
            #x_part[:, -1:1, -1:1 :] = x2_part[:, -1:1, -1:1 :]
            #x_part[:, -1:1, -1:1 :] = x[:, -(xs+1):-(xs+patch_scale-1), -(ys+1):-(ys+patch_scale-1), :]
            #x[:, xs+1:xs+patch_scale-1, ys+1:ys+patch_scale-1, :] = x_part[:, 1:-1, 1:-1, :]
            #x[:, -(xs+1):-(xs+patch_scale-1), -(ys+1):-(ys+patch_scale-1), :] = x2[:, -(xs+1):-(xs+patch_scale-1), -(ys+1):-(ys+patch_scale-1), :]
            #x[...,3:] += x2
        x = x2.clone()
        x[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :] = x_part
        #print(torch.max(x[:, xs:(xs+patch_scale), ys:(ys+patch_scale), 3:]))
        #print(x[:, xs:(xs+patch_scale), ys:(ys+patch_scale), 3:])
        #print(x[...,3:])
        #x[:, xs+1:xs+patch_scale-1, ys+1:ys+patch_scale-1, :] = x_part[:, 1:-1, 1:-1, :]
        return x

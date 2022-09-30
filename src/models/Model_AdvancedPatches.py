import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.Model_BasicNCA import BasicNCA

class AdvancedPatchesNCA(BasicNCA):
    def __init__(self, channel_n, fire_rate, device, checkCellsAlive=False, hidden_size=128):
        super(AdvancedPatchesNCA, self).__init__(channel_n, fire_rate, device, hidden_size)

    def forward_normal(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x[...,3:] = self.update(x, fire_rate, angle)[...,3:].detach()
        return x

    def forward(self, x, patch_scale=64, steps=1, fire_rate=None, angle=0.0):

        # Size of padding

        min_x = int((x.shape[1]/2) - (patch_scale/2))
        min_y = int((x.shape[2]/2) - (patch_scale/2))

        #print("BEGIN")
        #print(x.shape)

        x_part_img = x[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :].clone()
        x_part = x[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :].clone()

        #print(x_part.shape)

        #print(steps)

        x2  = x.clone()
        for step in range(steps):
           
            with torch.no_grad():
                x2 = self.update(x2, fire_rate, angle)
                x2 = torch.concat((x[...,:3], x2[...,3:]), 3)

            x_part = self.update(x_part, fire_rate, angle)
            x_part = torch.concat((x_part_img[...,:3], x_part[...,3:]), 3)
            #print(x_part.shape)


            x2_part = x2[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :].clone()
            x2_part[:, 1:-1, 1:-1 :] = x_part[:, 1:-1, 1:-1 :]
            x_part = x2_part
            x2[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :] = x_part#.detach()
            
        x = x2.clone()
        x[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :] = x_part

        return x

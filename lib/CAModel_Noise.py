import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.CAModel import CAModel

class CAModel_Noise(CAModel):
    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            img = x[...,:3]
            #noise = torch.zeros((x[...,3:]).shape, dtype=torch.float64)
            noise = torch.cuda.FloatTensor(*(x[...,:3]).shape).normal_() #torch.randn((x[...,:3]).shape, dtype=torch.float64)
            # uniform
            #noise = noise.to(self.device)
            x[...,:3] = x[...,:3] + 0.1 * noise
            x[...,3:] = self.update(x, fire_rate, angle)[...,3:]
            x[...,:3] = img
        return x

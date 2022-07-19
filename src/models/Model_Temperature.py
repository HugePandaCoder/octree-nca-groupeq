import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.Model_BasicNCA import BasicNCA
from src.agents.Agent_NCA import Agent
 
class TemperatureNCA(BasicNCA):
    def __init__(self, *args, **kwargs): #channel_n, fire_rate, device, hidden_size=128):
        super(TemperatureNCA, self).__init__(*args, **kwargs) #channel_n, fire_rate, device, hidden_size=128).__init__()

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        if False:
            self.device = device
            self.channel_n = channel_n

            self.fc0 = nn.Linear(channel_n*3, hidden_size)
            self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
            with torch.no_grad():
                self.fc1.weight.zero_()

            self.fire_rate = fire_rate
            self.to(self.device)

    # Source: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        #print(logits.shape)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1), logits.size(2), logits.size(3))
        return logits / temperature

    def temperature_scale_2(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        #print(logits.shape)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def forward(self, *args, **kwargs): # self, x, steps=1, fire_rate=None, angle=0.0):
        x = super().forward(*args, **kwargs)
        return self.temperature_scale(x)
        
        if False:
            for step in range(steps):
                x2 = self.update(x, fire_rate, angle).clone() #[...,3:][...,3:]
                x = torch.concat((x[...,:3], x2[...,3:]), 3)
            return x
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lib.CAModel import CAModel

class CAModel_Noise(CAModel):
    """.. note:: Test purposes"""
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



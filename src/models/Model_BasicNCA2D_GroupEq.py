import torch
import torch.nn as nn
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

class BasicNCA2D_GroupEq(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, normalization="batch"):
        super().__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.fire_rate = fire_rate
        padding = int((kernel_size-1) / 2)



    

        # Because group equivariant convolutions expand channels by 8
        base_hidden_size = hidden_size // 8

        # Lift Z2 input to P4M space

    
        self.perceive = P4ConvZ2(in_channels=channel_n, out_channels=base_hidden_size, kernel_size=kernel_size, padding=padding)

        # P4M → P4M convolutions
        self.update_conv = P4ConvP4(in_channels=base_hidden_size, out_channels=channel_n, kernel_size=1)

        # Normalization
        if normalization == "batch":
            self.bn = nn.BatchNorm3d(base_hidden_size, track_running_stats=False)
        elif normalization == "layer":
            self.bn = nn.LayerNorm([base_hidden_size, 1, 1])
        elif normalization == "group":
            self.bn = nn.GroupNorm(1, base_hidden_size)
        elif normalization == "none":
            self.bn = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type {normalization}")

        self.to(self.device)

    def update(self, x_in, fire_rate=None):
        # x_in: BHWC
        x = x_in.permute(0, 3, 1, 2)  # BCHW

        # Lift to P4M (B x C x H x W) → (B x 8C x H x W)
        y = self.perceive(x)  # B x (8*out) x H x W
        y = self.bn(y)
        y = F.relu(y)
        dy = self.update_conv(y)  # B x channel_n x H x W

        dy = dy.mean(dim=2)  # Reduce over group dimension → [B, channel_n, H, W] 






        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = (torch.rand_like(dy[:, :1, :, :]) > fire_rate).float().to(self.device)
        dy = dy * stochastic
        x = x + dy

        x = x.permute(0, 2, 3, 1)  # back to BHWC

        return x

    def forward(self, x, steps=10, fire_rate=0.5, visualize=False):
        if visualize:
            gallery = []
        for step in range(steps):
            x2 = self.update(x, fire_rate)
            x = torch.cat((x[..., :self.input_channels], x2[..., self.input_channels:]), dim=-1)
            if visualize:
                gallery.append(x.detach().cpu())
        if visualize:
            return x, gallery
        return x

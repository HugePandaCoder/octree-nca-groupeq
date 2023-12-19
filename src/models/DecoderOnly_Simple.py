import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnly_Simple(nn.Module):
    def __init__(self, batch_size, extra_channels, device=torch.device("cuda:0"), hidden_size=512):
        super(DecoderOnly_Simple, self).__init__()

        self.batch_size = batch_size
        self.extra_channels = extra_channels
        self.device = device
        self.hidden_size = hidden_size

        self.drop0 = nn.Dropout(0.25)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(extra_channels//2, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 128, 128]
            nn.Sigmoid()
        )

        self.list_backpropTrick = []
        for i in range(batch_size):
            self.list_backpropTrick.append(nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, x_vec_in, **kwargs):

        x_vec_in = x_vec_in.to(self.device)[:, :, None]
        #x_vec_in = self.drop0(x_vec_in)
        batch_emb = []
        for i in range(x_vec_in.shape[0]):
            batch_emb.append(self.list_backpropTrick[i].to(self.device)(x_vec_in[i:i+1]))
        #emb = torch.stack(batch_emb, dim=0)
        emb = torch.squeeze(torch.stack(batch_emb, dim=0))
        if x_vec_in.shape[0] == 1:
            emb = emb.to(self.device)[None, :]
        emb =  self.reparameterize(emb[:, :emb.shape[1]//2], emb[:, emb.shape[1]//2:])

        x = self.decoder(emb).transpose(1,3)
        
        #x = self.decoder(x).transpose(1,3)
        #x = x*2 -1
        return x

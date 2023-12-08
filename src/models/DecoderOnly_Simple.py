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
        self.vec_dec = nn.Linear(extra_channels, hidden_size*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size//2, kernel_size=4, stride=2, padding=1),  # [batch, 128, 8, 8]
            nn.ReLU(),
            #nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
            #nn.ReLU(),
            #nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
            #nn.ReLU(),
            nn.ConvTranspose2d(hidden_size//2, hidden_size//4, kernel_size=4, stride=2, padding=1),  # [batch, 64, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
            #nn.ReLU(),
            #nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
            #nn.ReLU(),
            nn.ConvTranspose2d(hidden_size//4, hidden_size//8, kernel_size=4, stride=2, padding=1),  # [batch, 32, 32, 32]
            nn.ReLU(),
            #nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
            #nn.ReLU(),
            #nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
            #nn.ReLU(),
            nn.ConvTranspose2d(hidden_size//8, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 64, 64]
            #nn.Sigmoid()  # Use sigmoid for normalizing outputs between 0 and 1
        )

        self.list_backpropTrick = []
        for i in range(batch_size):
            self.list_backpropTrick.append(nn.Conv1d(extra_channels, extra_channels, kernel_size=1, stride=1, padding=0, groups=extra_channels))

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

        x = self.vec_dec(emb)
        x = x.view(x.shape[0], self.hidden_size, 4, 4)
        #x = self.encoder(x)
        x = self.decoder(x).transpose(1,3)
        #x = x*2 -1
        return x

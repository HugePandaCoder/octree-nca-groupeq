import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderOnly(nn.Module):
    def __init__(self, batch_size, extra_channels, device=torch.device("cuda:0"), hidden_size=512):
        super(DecoderOnly, self).__init__()

        self.batch_size = batch_size
        self.extra_channels = extra_channels
        self.device = device
        self.hidden_size = hidden_size

        self.drop0 = nn.Dropout(0.25)

        # Decoder
        
        if False:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, hidden_size//2, kernel_size=4, stride=2, padding=1),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//2, hidden_size//2, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_size//2, hidden_size//4, kernel_size=4, stride=2, padding=1),  # [batch, 64, 16, 16]
                nn.ReLU(),
                nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//4, hidden_size//4, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_size//4, hidden_size//8, kernel_size=4, stride=2, padding=1),  # [batch, 32, 32, 32]
                nn.ReLU(),
                nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.Conv2d(hidden_size//8, hidden_size//8, kernel_size=1, stride=1, padding=0),  # [batch, 128, 8, 8]
                nn.ReLU(),
                nn.ConvTranspose2d(hidden_size//8, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 64, 64]
                #nn.Sigmoid()  # Use sigmoid for normalizing outputs between 0 and 1
            )


        # EMBEDDINGS
        self.vec_dec = nn.Linear(extra_channels, hidden_size*4*4)

        self.relu =  nn.ReLU()
        self.convt0 = nn.ConvTranspose2d(hidden_size, hidden_size//2, kernel_size=4, stride=2, padding=1)
        self.convt1 = nn.ConvTranspose2d(hidden_size//2, hidden_size//4, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(hidden_size//4, hidden_size//8, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(hidden_size//8, 3, kernel_size=4, stride=2, padding=1)

        self.conv0 = nn.Conv2d(hidden_size+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(hidden_size//2+extra_channels, hidden_size//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_size//4+extra_channels, hidden_size//4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_size//8+extra_channels, hidden_size//8, kernel_size=1, stride=1, padding=0)


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
        dx = x.view(x.shape[0], self.hidden_size, 4, 4)
        #x = self.encoder(x)
        #x = self.decoder(x).transpose(1,3)

        dx = self.relu(dx)
        #print(emb.shape, dx.shape, emb[:, :, None, None].repeat(1, 1, 4, 4).shape)
        dx = self.conv0(torch.cat((dx, emb[:, :, None, None].repeat(1, 1, 4, 4)), dim=1))
        dx = self.relu(dx)
        dx = self.convt0(dx)
        dx = self.relu(dx)

        dx = self.conv1(torch.cat((dx, emb[:, :, None, None].repeat(1, 1, 8, 8)), dim=1))
        dx = self.relu(dx)
        dx = self.convt1(dx)
        dx = self.relu(dx)

        dx = self.conv2(torch.cat((dx, emb[:, :, None, None].repeat(1, 1, 16, 16)), dim=1))
        dx = self.relu(dx)
        dx = self.convt2(dx)
        dx = self.relu(dx)

        dx = self.conv3(torch.cat((dx, emb[:, :, None, None].repeat(1, 1, 32, 32)), dim=1))
        dx = self.relu(dx)
        dx = self.convt3(dx)

        #x = x*2 -1
        return dx.transpose(1,3)
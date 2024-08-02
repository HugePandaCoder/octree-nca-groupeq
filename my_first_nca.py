import torch, einops
import torch.nn as nn
import torch.nn.functional as F
import timeit, time, tqdm
class MyNCA2D(nn.Module):
    def __init__(self, num_channels: int, num_input_channels: int, num_classes: int,
                 hidden_size: int, fire_rate: float, num_steps: int):
        super(MyNCA2D, self).__init__()

        self.fc0 = nn.Conv2d(2 * num_channels, hidden_size, kernel_size=1)
        self.fc1 = nn.Conv2d(hidden_size, num_channels - num_input_channels, kernel_size=1, bias=False)

        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same', 
                              padding_mode="reflect", groups=num_channels)
        self.batch_norm = nn.BatchNorm2d(hidden_size, track_running_stats=False)

        with torch.no_grad():
            self.fc0.weight.zero_()

        self.fire_rate = fire_rate
        self.num_channels = num_channels
        self.num_input_channels = num_input_channels
        self.num_steps = num_steps
        self.num_classes = num_classes

    def update(self, state):
        # state.shape: BCHW
        delta_state = self.conv(state)
        delta_state = torch.cat([state, delta_state], dim=1)
        delta_state = self.fc0(delta_state)
        delta_state = self.batch_norm(delta_state)
        delta_state = F.relu(delta_state)
        delta_state = self.fc1(delta_state)

        stochastic = torch.rand((delta_state.shape[0], delta_state.shape[1], delta_state.shape[2], 1), 
                                device=delta_state.device)
        stochastic = stochastic > 1 - self.fire_rate
        stochastic = stochastic.float()
        delta_state = delta_state * stochastic

        return state[:, self.num_input_channels:] + delta_state
        


    def forward(self, x):
        # x.shape: BCHW
        state = torch.zeros(x.shape[0], self.num_channels - self.num_input_channels,
                            x.shape[2], x.shape[3], device=x.device)
        state = torch.cat([x, state], dim=1)
        for _ in range(self.num_steps):
            new_state = self.update(state)
            state = torch.cat([x, new_state], dim=1)
        return state[:, self.num_input_channels:self.num_input_channels + self.num_classes]


nca = MyNCA2D(num_channels=16, num_input_channels=3, num_classes=2, hidden_size=64, 
              fire_rate=0.5, num_steps=10)

optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)

print("num params:", sum(p.numel() for p in nca.parameters()))
def dummy_train():
    c = time.time()
    for _ in tqdm.tqdm(range(100)):
        x = torch.randn(1, 3, 320, 320) # create random 320x320 RGB image
        y = nca(x)
        y.mean().backward() # backpropagate
        optimizer.step()
        optimizer.zero_grad()
    print("time", time.time() - c)

dummy_train()

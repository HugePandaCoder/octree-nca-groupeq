import numpy as np
import torch
import matplotlib.pyplot as plt
import einops
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
from src.losses.LossFunctions import DiceBCELoss, DiceLoss
import matplotlib, tqdm

def create_sample(segmentation_padding: int):
    sample_width, sample_height = 16, 8
    sample = np.zeros((1, sample_width, sample_height, 1))
    segmentation = np.zeros((1, sample_width, sample_height, 1))
    bar_width = 1
    bar_height = 4
    x_start, y_start = np.random.randint(0, sample_width - bar_width), np.random.randint(0, sample_height - bar_height)
    sample[0, x_start:x_start + bar_width, y_start:y_start + bar_height, 0] = 1
    segmentation[0, max(x_start - segmentation_padding, 0):min(x_start + bar_width + segmentation_padding, 16),
                max(y_start - segmentation_padding, 0):min(y_start + bar_height + segmentation_padding, 16)] = 1
    return sample, segmentation

sample, segmentation = create_sample(0)
#plt.imshow(sample[0, :, :, 0], cmap='gray')
#plt.imshow(segmentation[0, :, :, 0], cmap='jet', alpha=0.5)
#plt.show()

model = OctreeNCAV2(channel_n=4, fire_rate=0.5, 
                          device="cuda:0", hidden_size=24, 
                          input_channels=1, output_channels=1, 
                          steps=None, 
                          #octree_res_and_steps=[((16,8), 5), ((8,8), 5), ((4,4), 5), ((2,2), 5)],
                          octree_res_and_steps=[((16,8), 5)],
                          separate_models=False, compile=False, kernel_size=3)
model = model.to(model.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0016, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
dice_bce_loss = DiceBCELoss()
dice_loss = DiceLoss(useSigmoid=False)
loss_function = torch.nn.MSELoss()

for epoch in range(2000):

    losses = []
    for iteration in tqdm.tqdm(range(100)):
        sample_batch, segmentation_batch = [], []
        for b in range(3):
            sample, segmentation = create_sample(0)
            sample = einops.rearrange(sample, 'b h w c -> b c h w')
            segmentation = einops.rearrange(segmentation, 'b h w c -> b c h w')
            sample = torch.tensor(sample, dtype=torch.float32, device=model.device)
            segmentation = torch.tensor(segmentation, dtype=torch.float32, device=model.device)
            sample_batch.append(sample)
            segmentation_batch.append(segmentation)
        sample = torch.cat(sample_batch, dim=0)
        segmentation = torch.cat(segmentation_batch, dim=0)
        optimizer.zero_grad()
        out = model(sample, segmentation)


        #out = (einops.rearrange(sample, 'b c h w -> b h w c'), out[1])
        #print(out[0].shape, out[1].shape)
        #print(torch.all(out[0] == out[1]))
        #f, axarr = plt.subplots(1, 2)
        #axarr[0].imshow(out[0].detach().cpu().numpy()[0, :, :], cmap='gray')
        #axarr[1].imshow(out[1].detach().cpu().numpy()[0, :, :, :], cmap='gray')
        #plt.show()

        loss = loss_function(out[0], out[1])
        
        #print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item()   )
    scheduler.step()
    print(f"Epoch {epoch} Loss: {np.mean(losses):.4}, lr: {scheduler.get_last_lr()[0]}")


    if epoch % 10 == 0:
        with torch.no_grad():
            sample, segmentation = create_sample(0)
            sample = einops.rearrange(sample, 'b h w c -> b c h w')
            segmentation = einops.rearrange(segmentation, 'b h w c -> b c h w')
            sample = torch.tensor(sample, dtype=torch.float32)
            segmentation = torch.tensor(segmentation, dtype=torch.float32)
            out = model(sample, segmentation)
            print("dice loss:", dice_loss(out[0], out[1], binarize=True).item())

            f, axarr = plt.subplots(1, 2)

            axarr[0].imshow(sample[0].detach().cpu().numpy()[0, :, :], cmap='gray')
            axarr[1].imshow(out[0].detach().cpu().numpy()[0, :, :, :], cmap='gray')
            plt.show()
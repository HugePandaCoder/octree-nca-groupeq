import torch
import torch.nn as nn
from src.models.Model_BasicNCA3D import BasicNCA3D
import torchio as tio
import random
import math

class M3DNCA(nn.Module):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, scale_factor=4, levels=2, kernel_size=7):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(M3DNCA, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.scale_factor = scale_factor
        self.levels = levels

        self.model = nn.ModuleList()
        for i in range(self.levels):
            kernel_size = kernel_size if i == 0 else 3
            self.model.append(BasicNCA3D(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels))

    def make_seed(self, x):
        seed = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.channel_n), dtype=torch.float32, device=self.device)
        seed[..., 0:x.shape[-1]] = x 
        return seed

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        x = self.make_seed(x).to(self.device)
        #x = x.transpose(1,4)
        #y = y.transpose(1,4)
        #print(x.shape, y.shape)
        y = y.to(self.device)
        if self.training:
            x, y = self.forward_train(x, y)
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    def downscale_image(self, inputs, targets, iterations=1):
        max_pool = torch.nn.MaxPool3d(2, 2, 0)

        for i in range(iterations*int(math.log2(self.scale_factor))): # Multiples of 2 rightnow
            inputs = inputs.transpose(1,4)
            inputs = max_pool(inputs)
            inputs = inputs.transpose(1,4)
            targets = targets.transpose(1,4)
            targets = max_pool(targets)
            targets = targets.transpose(1,4)

        return inputs, targets

    def get_inference_steps(self, level=0):
        if isinstance(self.steps , list):
            return self.steps[level]
        return self.steps

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        max_pool = torch.nn.MaxPool3d(2, 2, 0)
        inputs_loc, targets_loc = self.downscale_image(x, y, iterations=self.levels-1)

        full_res, full_res_gt = x, y

        # For number of downscaling levels
        for m in range(self.levels): 
            # If last step -> run normal inference on final patch
            if m == self.levels-1:
                outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)
            else:
                # Create higher res image for next level -> Replace with single downscaling step
                next_res = full_res
                for i in range(self.levels - (m+2)):
                    next_res = next_res.transpose(1,4)
                    next_res = max_pool(next_res)
                    next_res = next_res.transpose(1,4)
                # Create higher res groundtruth for next level -> Replace with single downscaling step
                next_res_gt = full_res_gt
                for i in range(self.levels - (m+2)):
                    next_res_gt = next_res_gt.transpose(1,4)
                    next_res_gt = max_pool(next_res_gt)
                    next_res_gt = next_res_gt.transpose(1,4)

                # Run model inference on patch
                outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)

                # Upscale lowres features to next level
                up = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
                outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                outputs = up(outputs)
                outputs = torch.permute(outputs, (0, 2, 3, 4, 1))        
                # Concat lowres features with higher res image
                inputs_loc = torch.concat((next_res[...,:self.input_channels], outputs[...,self.input_channels:]), 4)

                # Array to store intermediate states
                targets_loc = next_res_gt
                size = (x.shape[1]//int(math.pow(self.scale_factor, (self.levels-1))),
                        x.shape[2]//int(math.pow(self.scale_factor, (self.levels-1))),
                        x.shape[3]//int(math.pow(self.scale_factor, (self.levels-1))))
                inputs_loc_temp = inputs_loc
                targets_loc_temp = targets_loc

                # Array to store next states
                inputs_loc = torch.zeros((inputs_loc_temp.shape[0], size[0], size[1], size[2] , inputs_loc_temp.shape[4])).to(self.device)
                targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], size[2] , targets_loc_temp.shape[4])).to(self.device)
                full_res_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/self.scale_factor), int(full_res.shape[2]/self.scale_factor), int(full_res.shape[3]/self.scale_factor), full_res.shape[4])).to(self.device)
                full_res_gt_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/self.scale_factor), int(full_res.shape[2]/self.scale_factor), int(full_res.shape[3]/self.scale_factor), full_res_gt.shape[4])).to(self.device)

                # Scaling factors
                factor = self.levels - m -2
                factor_pow = math.pow(2, factor)
                #factor = self.levels - m -2
                #factor_pow = math.pow(self.scale_factor, factor)

                # Choose random patch of image for each element in batch
                for b in range(inputs_loc.shape[0]): 
                    while True:
                        pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                        pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])
                        pos_z = random.randint(0, inputs_loc_temp.shape[3] - size[2])
                        break

                    # Randomized start position for patch
                    pos_x_full = int(pos_x * factor_pow)
                    pos_y_full = int(pos_y * factor_pow)
                    pos_z_full = int(pos_z * factor_pow)
                    size_full = [int(full_res.shape[1]/self.scale_factor), int(full_res.shape[2]/self.scale_factor), int(full_res.shape[3]/self.scale_factor)]

                    # Set current patch of inputs and targets
                    inputs_loc[b] = inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                    if len(targets_loc.shape) > 4:
                        targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                    else:
                        targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]

                    # Update full res image to patch of full res image
                    full_res_new[b] = full_res[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]
                    full_res_gt_new[b] = full_res_gt[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]

                full_res = full_res_new
                full_res_gt = full_res_gt_new

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc
    
    def forward_eval(self, x: torch.Tensor):
        max_pool = torch.nn.MaxPool3d(2, 2, 0)
        inputs_loc, _ = self.downscale_image(x, x, iterations=self.levels-1)

        full_res = x

        with torch.no_grad():
            # Start with low res lvl and go to high res level
            for m in range(self.levels):
                if m == self.levels-1:
                    outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)
                # Scale m-1 times 
                else:
                    up = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
                    outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)

                    # Upscale lowres features to next level
                    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                    outputs = up(outputs)
                    inputs_loc = x     
                    outputs = torch.permute(outputs, (0, 2, 3, 4, 1))         

                    # Create higher res image for next level -> Replace with single downscaling step
                    next_res = full_res
                    for i in range(self.levels - (m +2)):
                        next_res = next_res.transpose(1,4)
                        next_res = max_pool(next_res)
                        next_res = next_res.transpose(1,4)

                    # Concat lowres features with higher res image
                    inputs_loc = torch.concat((next_res[...,:self.input_channels], outputs[...,self.input_channels:]), 4)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels]
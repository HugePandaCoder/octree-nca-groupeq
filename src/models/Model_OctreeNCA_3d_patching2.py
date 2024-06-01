from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.models.Model_BasicNCA3D import BasicNCA3D
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp

from src.models.Model_OctreeNCA_3D import OctreeNCA3D
import matplotlib.pyplot as plt

class OctreeNCA3DPatch2(OctreeNCA3D):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, 
                 scale_factor=None, levels=None, kernel_size=None,
                 octree_res_and_steps: list=None, separate_models: bool=False,
                 compile: bool=False,
                 patch_sizes=None):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCA3DPatch2, self).__init__(channel_n, fire_rate, device, steps, hidden_size, input_channels, output_channels, scale_factor, levels, kernel_size, octree_res_and_steps, separate_models, compile)


        self.computed_upsampling_scales = []
        for i in range(len(self.octree_res)-1):
            t = []
            for c in range(3):
                t.append(self.octree_res[i][c]//self.octree_res[i+1][c])
            self.computed_upsampling_scales.append(np.array(t).reshape(1, 3))

        self.patch_sizes = patch_sizes

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        #x: BHWDC
        #y: BHWDC

        #if y is not None:
        #    y = y.to(self.device)

        if self.training:
            if batch_duplication != 1:
                x = torch.cat([x] * batch_duplication, dim=0)
                y = torch.cat([y] * batch_duplication, dim=0)

            x, y = self.forward_train(x, y)
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    @torch.no_grad()
    def downscale(self, x: torch.Tensor, level: int):
        x = self.align_tensor_to(x, "BCHWD")
        self.remove_names(x)

        out = F.interpolate(x, size=self.octree_res[level])
        out.names = ('B', 'C', 'H', 'W', 'D')
        x.names = ('B', 'C', 'H', 'W', 'D')
        return out
    
    def remove_names(self, x: torch.Tensor):
        x.names = [None, None, None, None, None]

    def align_tensor_to(self, x: torch.Tensor, to: str) -> torch.Tensor:
        assert x.names == ('B', 'H', 'W', 'D', 'C') or \
                x.names == ('B', 'C', 'H', 'W', 'D'), f"Expected names ('B', 'H', 'W', 'D', 'C') or ('B', 'C', 'H', 'W', 'D'), got {x.names}"
        if to == "BCHWD":
            if x.names == ('B', 'H', 'W', 'D', 'C'):
                self.remove_names(x)
                x = x.permute(0, 4, 1, 2, 3)
                x.names = ('B', 'C', 'H', 'W', 'D')
                return x
            elif x.names == ('B', 'C', 'H', 'W', 'D'):
                return x
        elif to == "BHWDC":
            if x.names == ('B', 'C', 'H', 'W', 'D'):
                self.remove_names(x)
                x = x.permute(0, 2, 3, 4, 1)
                x.names = ('B', 'H', 'W', 'D', 'C')
                return x
            elif x.names == ('B', 'H', 'W', 'D', 'C'):
                return x
        assert False, f"Expected to be aligned to BCHWD or BHWDC, got {to}"


    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        original = x.permute(0, 4, 1, 2, 3)
        x.names = ('B', 'H', 'W', 'D', 'C')
        y.names = ('B', 'H', 'W', 'D', 'C')
        original.names = ('B', 'C', 'H', 'W', 'D')
        #x: BHWDC
        #y: BHWDC

        if self.patch_sizes[-1] is not None:
            x_new = torch.zeros(x.shape[0], *self.patch_sizes[-1], self.channel_n,
                                dtype=torch.float, device=self.device, 
                                names=('B', 'H', 'W', 'D', 'C'))
            current_patch = np.zeros((x.shape[0], 2, 3), dtype=int)
            x = self.downscale(x, -1)
            x = self.align_tensor_to(x, "BHWDC")
            self.remove_names(x_new)
            self.remove_names(x)
            for b in range(x.shape[0]):
                h_start = self.my_rand_int(0, self.octree_res[-1][0]-self.patch_sizes[-1][0])
                w_start = self.my_rand_int(0, self.octree_res[-1][1]-self.patch_sizes[-1][1])
                d_start = self.my_rand_int(0, self.octree_res[-1][2]-self.patch_sizes[-1][2])
                current_patch[b] = np.array([[h_start, w_start, d_start], 
                                        [self.patch_sizes[-1][0] + h_start, 
                                        self.patch_sizes[-1][1] + w_start, 
                                        self.patch_sizes[-1][2] + d_start]
                                        ])
                
                x_new[b,:,:,:, :self.input_channels] = \
                x[b,    current_patch[b,0,0]:current_patch[b,1,0],
                        current_patch[b,0,1]:current_patch[b,1,1],
                        current_patch[b,0,2]:current_patch[b,1,2], :]
            x_new.names = ('B', 'H', 'W', 'D', 'C')
            x = x_new
        else:
            x_new = torch.zeros(x.shape[0], *self.octree_res[-1], self.channel_n,
                                dtype=torch.float, device=self.device)
            current_patch = np.array([[[0,0,0], [*self.octree_res[-1]]]] * x.shape[0])
            x = self.downscale(x, -1)
            x = self.align_tensor_to(x, "BHWDC")
            self.remove_names(x)
            x_new[:,:,:,:, :self.input_channels] = x
            x = x_new
            x.names = ('B', 'H', 'W', 'D', 'C')

        #x: BHWDC


        for level in range(len(self.octree_res)-1, -1, -1):

            x = self.align_tensor_to(x, "BHWDC")
            self.remove_names(x)

            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            x.names = ('B', 'H', 'W', 'D', 'C')


            if level > 0:
                #upscale states
                x = self.align_tensor_to(x, "BCHWD")
                self.remove_names(x)
                x = torch.nn.Upsample(scale_factor=tuple(self.computed_upsampling_scales[level-1][0]), 
                                      mode='nearest')(x)
                current_patch *= self.computed_upsampling_scales[level-1]
            
                original_right_resolution = self.downscale(original, level-1)
                assert original_right_resolution.names == ('B', 'C', 'H', 'W', 'D')
                self.remove_names(original_right_resolution)
                #cut out patch from input_channels
                if self.patch_sizes[level-1] is not None:
                    x_new = torch.zeros(x.shape[0], self.channel_n, *self.patch_sizes[level-1], device=self.device, dtype=torch.float)
                    for b in range(x.shape[0]):
                        h_offset = self.my_rand_int(0, x.shape[2]-self.patch_sizes[level-1][0])
                        w_offset = self.my_rand_int(0, x.shape[3]-self.patch_sizes[level-1][1])
                        d_offset = self.my_rand_int(0, x.shape[4]-self.patch_sizes[level-1][2])

                        current_patch[b, 0] += np.array([h_offset, w_offset, d_offset])
                        current_patch[b, 1] = current_patch[b, 0] + np.array(self.patch_sizes[level-1])
                        
                        x_new[b, :self.input_channels] = original_right_resolution[b, :,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1],
                                        current_patch[b,0,2]:current_patch[b,1,2]]
                        
                        x_new[b, self.input_channels:] = x[b, self.input_channels:,
                                        h_offset:h_offset + self.patch_sizes[level-1][0],
                                        w_offset:w_offset + self.patch_sizes[level-1][1],
                                        d_offset:d_offset + self.patch_sizes[level-1][2]]
                    x = x_new
                else:
                    for b in range(x.shape[0]):
                        x[b, :self.input_channels] = original_right_resolution[b, :,
                                        current_patch[b,0,0]:current_patch[b,1,0],
                                        current_patch[b,0,1]:current_patch[b,1,1],
                                        current_patch[b,0,2]:current_patch[b,1,2]]
                x.names = ('B', 'C', 'H', 'W', 'D')
        
        #x: BHWDC


        y_new = torch.zeros(y.shape[0], x.shape[1], x.shape[2], x.shape[3],
                             y.shape[4], device=self.device, dtype=torch.float)
        for b in range(x.shape[0]):
            y_new[b] = y[b, current_patch[b,0,0]:current_patch[b,1,0],
                        current_patch[b,0,1]:current_patch[b,1,1],
                        current_patch[b,0,2]:current_patch[b,1,2], :] 
        y = y_new
        
        self.remove_names(x)
        x = x[..., self.input_channels:self.input_channels+self.output_channels]

        return x, y
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        temp = self.patch_sizes
        self.patch_sizes = [None] * len(self.patch_sizes)
        out, _ = self.forward_train(x, x)
        self.patch_sizes = temp
        return out
    
    def create_inference_series(self, x: torch.Tensor, steps=None):
        assert False, "Not implemented yet"
        #x: BCHWD
        x = x.permute(0, 2,3,4, 1)
        #x: BHWDC
        x = self.make_seed(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x: BCHWD
        x = x.to(self.device)
        lod = Octree3DNoStates(x, self.octree_res)
        
        inference_series = [] #list of BHWDC tensors

        for level in list(range(len(lod.levels_of_detail)))[::-1]:
            x = lod.levels_of_detail[level]
            #x: BCHWD
            x = x.permute(0, 2,3,4, 1)
            #x: BHWDC
            inference_series.append(x)
            
            if self.separate_models:
                x = self.backbone_ncas[level](x, steps=self.inference_steps[level], fire_rate=self.fire_rate)
            else:
                x = self.backbone_nca(x, steps=self.inference_steps[level], fire_rate=self.fire_rate)

            inference_series.append(x)
            #x: BHWDC
            x = x.permute(0, 4, 1, 2, 3)
            # x: BCHWD

            lod.levels_of_detail[level] = x
            if level > 0:
                lod.upscale_states(level)

        outputs = lod.levels_of_detail[0]
        return inference_series
    
    def my_rand_int(self, low, high):
        if high == low:
            return low
        return random.randint(low, high)
        #return np.random.randint(low, high)
    

class Octree3DNoStates:
    @torch.no_grad()
    def __init__(self, init_batch: torch.Tensor, octree_res: list[int]) -> None:

        assert init_batch.ndim == 5, f"init_batch must be BCHWD tensor, got shape {init_batch.shape}"
        
        self.levels_of_detail = [init_batch]
        assert init_batch.shape[2:] == octree_res[0], f"init_batch must have shape {octree_res[0]}, got shape {init_batch.shape[2:]}"

        for resolution in octree_res[1:]:
            lower_res = F.interpolate(self.levels_of_detail[-1], size=resolution)
            self.levels_of_detail.append(lower_res)
                                     

    def plot(self, output_path: str = 'octree.pdf') -> None:
        fig, axs = plt.subplots(1, len(self.levels_of_detail), figsize=(20, 20))
        for i, img in enumerate(self.levels_of_detail):
            depth = img.shape[4]
            axs[i].imshow(img[0, 0, :, :, depth//2].cpu(), cmap='gray')
        plt.savefig(output_path, bbox_inches='tight')
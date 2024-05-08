from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from src.models.Model_BasicNCA3D import BasicNCA3D
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp

class OctreeNCA3D(nn.Module):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, 
                 scale_factor=None, levels=None, kernel_size=None):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(OctreeNCA3D, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.scale_factor = scale_factor
        self.levels = levels
        self.fast_inf = False
        self.margin = 20

        self.backbone_nca = BasicNCA3D(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels, kernel_size=3)

    def make_seed(self, x):
        # x: BHWDC
        seed = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.channel_n), dtype=torch.float32, device=self.device)
        seed[..., 0:x.shape[-1]] = x 
        # seed: BHWDC 
        return seed


    def get_gpu_memory(self):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        # print(memory_use_values)
        return memory_use_values

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        #x: BHWDC
        #y: BHWDC
        
        #show image and save
        #plt.imshow(x[0,:,:,31,:].cpu())
        #plt.savefig("temp.pdf")


        x = self.make_seed(x).to(self.device)
        #x: BHWDC
        x = x.permute(0, 4, 1, 2, 3)
        # x: BCHWD
        x = x.to(self.device)

        if y is not None:
            y = y.to(self.device)

        if self.training:
            if batch_duplication != 1:
                x = torch.cat([x] * batch_duplication, dim=0)
                y = torch.cat([y] * batch_duplication, dim=0)

            x, y = self.forward_train(x, y)
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        lod = Octree3D(x, self.input_channels)
        #lod.plot("hippocampus_octree.pdf")
        #exit()

        #print("LOD")
        #[print(x.shape) for x in lod.levels_of_detail]
        #print("LOD")


        steps_arr = [2] * len(lod.levels_of_detail)
        steps_arr[-1] = 16

        
        for level in list(range(len(lod.levels_of_detail)))[::-1]: #micro to macro (low res to high res)
            x = lod.levels_of_detail[level]
            #x: BCHWD
            x = x.permute(0, 2,3,4, 1)
            #x: BHWDC

            x = self.backbone_nca(x, steps=steps_arr[level], fire_rate=self.fire_rate)
            #x: BHWDC
            x = x.permute(0, 4, 1, 2, 3)
            # x: BCHWD

            lod.levels_of_detail[level] = x


            if level > 0:
                lod.upscale_states(level)
        
        outputs = lod.levels_of_detail[0].permute(0, 2,3,4, 1)
        #outputs: BHWDC

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], y
    
    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor):
        out, _ = self.forward_train(x, x)
        return out
    

class Octree3D:
    @torch.no_grad()
    def __init__(self, init_batch: torch.Tensor, input_channels: int) -> None:
        self.input_channels = input_channels

        assert init_batch.ndim == 5, f"init_batch must be BCHWD tensor, got shape {init_batch.shape}"
        #assert init_batch.shape[2] == init_batch.shape[3], f"init_batch must be square, got shape {init_batch.shape}"
        #assert init_batch.shape[3] == init_batch.shape[4], f"init_batch must be square, got shape {init_batch.shape}"
        
        self.levels_of_detail = [init_batch]
        while self.levels_of_detail[-1].shape[2] > 16 or self.levels_of_detail[-1].shape[3] > 16 or self.levels_of_detail[-1].shape[4] > 16:
            lower_res = F.avg_pool3d(self.levels_of_detail[-1], 2)
            self.levels_of_detail.append(lower_res)

    def plot(self, output_path: str = 'octree.pdf') -> None:
        fig, axs = plt.subplots(1, len(self.levels_of_detail), figsize=(20, 20))
        for i, img in enumerate(self.levels_of_detail):
            depth = img.shape[4]
            axs[i].imshow(img[0, 0, :, :, depth//2].cpu(), cmap='gray')
        plt.savefig(output_path, bbox_inches='tight')

    def upscale_states(self, from_level: int) -> None:
        assert from_level in range(1, len(self.levels_of_detail)), "from_level must be in range(1, len(levels_of_detail))"
        temp = self.levels_of_detail[from_level]
        #temp: BCHWD
        temp = temp[:, self.input_channels:]

        upsampled_states = torch.nn.Upsample(scale_factor=2, mode='nearest')(temp)

        self.levels_of_detail[from_level-1] = torch.cat([self.levels_of_detail[from_level-1][:, :self.input_channels], upsampled_states], dim=1)
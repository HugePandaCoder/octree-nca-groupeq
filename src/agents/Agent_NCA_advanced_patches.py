import torch
import numpy as np
from src.utils.helper import convert_image, dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent_NCA import Agent
from src.losses.LossFunctions import DiceLoss
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output
from src.utils.helper import dump_pickle_file, load_pickle_file
import os
from src.losses.LossFunctions import DiceLoss, DiceBCELoss
import copy

class Agent_AdvancedPatches(Agent):
    def get_outputs(self, data):
        r"""Get the outputs of the model
            Args:
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        outputs, targets = outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

        patch_scale = 64

        min_x = int((inputs.shape[1]/2) - (patch_scale/2))
        min_y = int((inputs.shape[2]/2) - (patch_scale/2))

        #print("BEGIN")
        #print(x.shape)

        outputs = outputs[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :].clone()
        targets = targets[:, min_x:(min_x+patch_scale), min_y:(min_y+patch_scale), :].clone()

        return outputs, targets
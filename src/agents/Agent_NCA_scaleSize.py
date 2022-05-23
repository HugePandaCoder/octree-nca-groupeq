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

class Agent_ScaleSize(Agent):
    def get_outputs(self, data):
        r"""Get the outputs of the model
            Args:
                data (int, tensor, tensor): id, inputs, targets
        """
        current_epoch = self.exp.currentStep
        input_size = copy.deepcopy(self.exp.get_from_config("input_size"))
        smaller = 60 - int(current_epoch/3)
        if smaller > 0:
            new_input_size = input_size
            new_input_size = (new_input_size[0] - smaller, new_input_size[1] - smaller)
            #new_input_size[1] = 
            #print(new_input_size)
            self.exp.dataset.set_size(new_input_size)
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach(), id)
        return outputs[..., 3:6], targets
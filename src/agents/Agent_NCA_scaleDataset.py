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

class Agent_ScaleDataset(Agent):

    def initialize_epoch(self):
        r"""Everything that should happen once before each epoch should be defined here.
        """
        if( self.exp.currentStep < 30):
            current_epoch = 0.1
        else:
            current_epoch = (self.exp.currentStep - 30) * 0.01
        subset = min(1, current_epoch)
        self.exp.set_model_state_subset('train', subset=subset)
        return

    def get_outputs(self, data):
        r"""Get the outputs of the model
            Args:
                data (int, tensor, tensor): id, inputs, targets
        """

        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach(), id)
        return outputs[..., 3:6], targets
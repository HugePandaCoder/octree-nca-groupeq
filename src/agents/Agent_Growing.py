import torch
import numpy as np
from src.utils.helper import convert_image, dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent import BaseAgent
from src.losses.LossFunctions import DiceLoss
import torch.optim as optim
from IPython.display import clear_output
from src.utils.helper import dump_pickle_file, load_pickle_file
import os
from src.agents.Agent_NCA import Agent

class Agent_Growing(Agent):
    def get_outputs(self, data, full_img=False):
        r"""Get the outputs of the model
            Args:
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., 0:4], targets
import torch
import numpy as np
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output
from src.agents.Agent import BaseAgent
from src.losses.LossFunctions import DiceLoss

class Agent(BaseAgent):

    def initialize(self):
        super().initialize()
        self.input_channels = self.exp.get_from_config('input_channels')
        self.output_channels = self.exp.get_from_config('output_channels')

    def prepare_data(self, data, eval=False):
        r"""Prepare the data to be used with the model
            Args:
                data (int, tensor, tensor): identity, image, target mask
            Returns:
                inputs (tensor): Input to model
                targets (tensor): Target of model
        """
        id, inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if self.exp.dataset.slice is None:
            inputs, targets = torch.unsqueeze(inputs, 1), torch.unsqueeze(targets, 1) 
        #print(inputs.shape)
        if len(inputs.shape) == 4:
            return id, inputs.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2)
        else:
            return id, inputs, targets

    def get_outputs(self, data, **kwargs):
        _, inputs, targets = data
        if len(inputs.shape) == 4:
            return (self.model(inputs)).permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        else:
            return (self.model(inputs)).permute(0, 2, 3, 4, 1), targets.permute(0, 2, 3, 4, 1)

    def prepare_image_for_display(self, image):
        return image #.permute(0, 2, 3, 1)

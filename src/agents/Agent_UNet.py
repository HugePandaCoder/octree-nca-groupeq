import torch
import numpy as np
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output
from src.agents.Agent import BaseAgent
from src.losses.LossFunctions import DiceLoss

class Agent(BaseAgent):

    r"""Prepare the data to be used with the model
        Args:
            data (int, tensor, tensor): identity, image, target mask
        Returns:
            inputs (tensor): Input to model
            targets (tensor): Target of model
    """
    def prepare_data(self, data, eval=False):
        id, inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        return id, inputs.permute(0, 3, 1, 2), targets.permute(0, 3, 1, 2)

    def get_outputs(self, data):
        _, inputs, targets = data
        #print(inputs.shape)
        return self.model(inputs), targets

    def prepare_image_for_display(self, image):
        return image.permute(0, 2, 3, 1)
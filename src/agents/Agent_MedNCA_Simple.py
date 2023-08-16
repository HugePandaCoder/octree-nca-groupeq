import torch
from src.agents.Agent_UNet import Agent

class MedNCAAgent(Agent):
    """Base agent for training UNet models
    """
    def initialize(self):
        super().initialize()

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        _, inputs, targets = data
        
        inputs, targets = self.model(inputs, targets)
        return inputs, targets
        #if len(inputs.shape) == 4:
        #    return (self.model(inputs)).permute(0, 2, 3, 1), targets.permute(0, 2, 3, 1)
        #else:
        #    return (self.model(inputs)).permute(0, 2, 3, 4, 1), targets #.permute(0, 2, 3, 4, 1)

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
from src.agents.Agent_NCA import Pool

class Agent_ScaleSize(Agent):
    def initialize_epoch(self):
        r"""Create a pool for the current epoch
        """
        if self.exp.get_from_config('Persistence'):
            self.epoch_pool = Pool()
        
        #self.exp.get_from_config('evaluate_interval')

        if( self.exp.currentStep %30 == 1 and self.exp.currentStep > 5 and self.exp.currentStep < 100):
            self.model = self.model.increaseHiddenLayer(increase=16*int(self.exp.currentStep/10), optimizer=self.optimizer)
           # self.model = self.model.doubleHiddenLayer() #
            #print(self.optimizer.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
            #self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))
            self.optimizer.zero_grad()

    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            Args:
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            Returns:
                loss item
        """
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        self.optimizer.zero_grad()
        #print(outputs.shape)
        #print(targets.shape)
        loss = loss_f(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def conclude_epoch(self):
        r"""Everything that should happen once after each epoch should be defined here.
        """
        return


        
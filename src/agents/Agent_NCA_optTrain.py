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

class Agent_OptTrain(Agent):

    def initialize(self):
        super().initialize()
        self.x_pos = 0
        self.y_pos = 0

    def initialize_epoch(self):
        r"""Create a pool for the current epoch
        """
        #if self.exp.get_from_config('Persistence'):
        #    self.epoch_pool = Pool()
        
        #self.exp.get_from_config('evaluate_interval')

        numberOfSteps = 20
        numberOfDoubling = 4

        if( self.exp.currentStep %numberOfSteps == 1 and self.exp.currentStep > 1 and self.exp.currentStep < numberOfDoubling*numberOfSteps+1):
            #self.model = self.model.increaseChannelSize(increase=16*int(self.exp.currentStep/10))
            #self.exp.config['channel_n'] = self.exp.config['channel_n']*2
            #self.exp.projectConfig[0]['channel_n'] = self.exp.config['channel_n']
            
            self.model = self.model.increaseHiddenLayer(increase=16*int(self.exp.currentStep/10), optimizer=self.optimizer)
           
           
           # self.model = self.model.doubleHiddenLayer() #
            #print(self.optimizer.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
            #self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))
            self.optimizer.zero_grad()

    #def initialize_epoch(self):
    #    r"""Everything that should happen once before each epoch should be defined here.
    #    """
    #    if( self.exp.currentStep < 30):
    #        current_epoch = 0.1
    #    else:
    #        current_epoch = (self.exp.currentStep - 30) * 0.01
    #    subset = min(1, current_epoch)
    #    self.exp.set_model_state_subset('train', subset=subset)
    #    return

    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            Args:
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            Returns:
                loss item
        """
        data = self.prepare_data(data)

        patch_scale = 64
        image_size = 256
        #if torch.randint(0,1) == 1:

        self.x_pos = (self.x_pos +1) % int(image_size/patch_scale)
        
        if self.x_pos == 0:
            self.y_pos = (self.y_pos +1) % int(image_size/patch_scale)

        xs = self.x_pos * patch_scale#torch.randint(0, image_size - patch_scale, (1,))[0] 
        ys = self.y_pos * patch_scale#torch.randint(0, image_size - patch_scale, (1,))[0] 

        #xs = torch.randint(0, image_size - patch_scale, (1,))[0] 
        #ys = torch.randint(0, image_size - patch_scale, (1,))[0] 

        outputs, targets = self.get_outputs(data, xs, ys, patch_scale)
        self.optimizer.zero_grad()
        #print("________________________")
        #print(targets.shape)
        outputs = outputs[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :]
        targets = targets[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :]
        #print(targets.shape)
        #print(torch.sum(targets))

        outputs = outputs[:, 1:-1, 1:-1 :]
        targets = targets[:, 1:-1, 1:-1 :]

        #loss_function = DiceBCELoss()

        #print(torch.max(outputs))
        #print(torch.max(targets))
        #print(torch.min(outputs))
        #print(torch.min(targets))

        loss = loss_f(outputs, targets) # + loss_function(outputs, targets)

        if torch.sum(targets) == 0: 
            loss = loss * 0.1

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def get_outputs(self, data, xs=-1, ys=-1, patch_scale=-1):
        r"""Get the outputs of the model
            Args:
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data
        if ys == -1:
            outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        else:
            outputs = self.model.forward_opt(inputs, xs, ys, patch_scale, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach(), id)
        return outputs[..., 3:6], targets
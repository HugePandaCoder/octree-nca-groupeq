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
        xs = torch.randint(0, 256 - patch_scale, (1,))[0] 
        ys = torch.randint(0, 256 - patch_scale, (1,))[0] 

        print("___________")
        print(xs)
        print(ys)

        outputs, targets = self.get_outputs(data, xs, ys, patch_scale)
        self.optimizer.zero_grad()

        outputs = outputs[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :]
        targets = targets[:, xs:(xs+patch_scale), ys:(ys+patch_scale), :]
        outputs = outputs[:, 1:-1, 1:-1 :]
        targets = targets[:, 1:-1, 1:-1 :]

        loss_function = DiceBCELoss()

        loss = loss_f(outputs, targets) + loss_function(outputs, targets)
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
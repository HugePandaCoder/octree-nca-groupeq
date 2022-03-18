import torch
import numpy as np
from src.utils.helper import convert_image
from src.agents.Agent import BaseAgent
from src.losses.LossFunctions import DiceLoss
import torch.optim as optim
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from IPython.display import clear_output


class Agent(BaseAgent):
    
    def initialize(self):
        super().initialize()
        self.pool = Pool()

    def loss_noOcillation(self, x, target, freeChange=True):
        #x = torch.flatten(x)
        if freeChange:
            x[x <= 1] = 0
            loss = x.sum() / torch.numel(x)
        else:
            xin_sum = torch.sum(x) + 1
            x = torch.square(target-x)
            loss = torch.sum(x) / xin_sum
        return loss

    r"""TODO 
    """
    def loss_f(self, x, target):
        return torch.mean(torch.pow(x[..., :3]-target, 2), [-2,-3,-1])

    r"""Creates a padding around the tensor 
        Args:
            target (tensor)
            padding (int): padding on all 4 sides
    """
    def pad_target_f(self, target, padding):
        target = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32)).to(self.device)
        return target

    r"""Create a seed for the NCA - TODO: Currently only 0 input
        Args:
            shape ([int, int]): height, width shape
            n_channels (int): Number of channels
    """
    def make_seed(self, img):
        seed = torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
        seed[..., :img.shape[3]] = img
        return seed

    r"""Repeat batch -> Useful for better generalisation when doing random activated neurons
        Args:
            seed (tensor): Seed for NCA
            target (tensor): Target of Model
            repeat_factor (int): How many times it should be repeated
    """
    def repeatBatch(self, seed, target, repeat_factor):
        return torch.Tensor.repeat_interleave(seed, repeat_factor, dim=0), torch.Tensor.repeat_interleave(target, repeat_factor, dim=0)

    r"""Get the number of steps for inference, if its set to an array its a random value inbetween
    """
    def getInferenceSteps(self):
        if len(self.exp.get_from_config('inference_steps')) == 2:
            steps = np.random.randint(self.exp.get_from_config('inference_steps')[0], self.exp.get_from_config('inference_steps')[1])
        else:
            steps = self.exp.get_from_config('inference_steps')[0]
        return steps

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
        inputs = self.make_seed(inputs)
        if not eval:
            if self.exp.get_from_config('Persistence'):
                inputs = self.pool.getFromPool(inputs, id, self.device)
            inputs, targets = self.repeatBatch(inputs, targets, self.exp.get_from_config('repeat_factor'))
        return id, inputs, targets

    r"""Get the outputs of the model
        Args:
            data (int, tensor, tensor): id, inputs, targets
    """
    def get_outputs(self, data):
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach(), id)
        return outputs[..., 3:6], targets

    r"""Create a pool for the current epoch
    """
    def initialize_epoch(self):
        if self.exp.get_from_config('Persistence'):
            self.epoch_pool = Pool()
        return

    r"""Set epoch pool as active pool
    """
    def conclude_epoch(self):
        if self.exp.get_from_config('Persistence'):
            self.pool = self.epoch_pool
            print("Pool_size: " + str(len(self.pool)))
        return

    def prepare_image_for_display(self, image):
        return image[...,0:3]

r"""Keeps the previous outputs of the model in stored in a pool
"""
class Pool():
    def __init__(self):
        self.pool = {}

    def __len__(self):
        return len(self.pool)

    r"""Add a value to the pool
        Args:
            output (tensor): Output to store
            idx (int): idx in dataset
            exp (Experiment): All experiment related configs
            dataset (Dataset): The dataset
    """
    def addToPool(self, outputs, ids):
        for i, key in enumerate(ids):
            self.pool[key] = outputs[i]

    r"""Get value from pool
        Args:
            item (int): idx of item
            dataset (Dataset)
    """
    def getFromPool(self, inputs, ids, device):   
        for i, key in enumerate(ids):
            if key in self.pool.keys():
                inputs[i] = self.pool[key].to(device)
        return inputs

    
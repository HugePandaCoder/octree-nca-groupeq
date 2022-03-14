from re import I
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Nii_Gz_Dataset import Nii_Gz_Dataset
from png_Dataset import png_Dataset
from IPython.display import clear_output
from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from LossFunctions import DiceLoss, DiceBCELoss
from Experiment import Experiment, DataSplit
from Agent_NCA import Agent
import sys

# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

config = {
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': "models/nca_test_c16_cf05_newResLoss_v1.pth",
    'reload': True,
    'device':"cuda:0",
    'n_epoch': 200,
    # Learning rate
    'lr': 2e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [64],
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 8,
    'repeat_factor': 2,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.5,
    'Persistence': True,
}

# Define Experiment
dataset = Nii_Gz_Dataset(config['input_size'])
exp = Experiment(config, dataset)
exp.set_model_state('train')

device = torch.device(config['device'])

# Create Model
ca = CAModel(config['channel_n'], config['cell_fire_rate'], device).to(device)
if config['reload'] == True:
    ca.load_state_dict(torch.load(config['model_path']))

loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()

agent = Agent(ca, config)
agent.train(dataset, config, loss_function)


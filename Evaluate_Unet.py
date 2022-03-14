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

from LossFunctions import DiceLoss
import nibabel as nib
import sys
import os

from Agent_UNet import Agent
from Experiment import Experiment, DataSplit

from unet import UNet2D

config = {
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': "models/unet_test2.pth",
    'reload': True,
    'device':"cuda:0",
    'n_epoch': 40,
    # Learning rate
    'lr': 2e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [128],
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 1,
    'persistence_chance':0.5,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
}

# Define Experiment
dataset = Nii_Gz_Dataset(config['input_size'])
exp = Experiment(config, dataset)
exp.set_model_state('test')

device = torch.device("cpu")

model = UNet2D(in_channels=3, padding=1, out_classes=3).to(device)
model.load_state_dict(torch.load(config['model_path']))

diceLoss = DiceLoss(useSigmoid=True)

agent = Agent(model, config)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

agent.test(dataset, config, diceLoss)

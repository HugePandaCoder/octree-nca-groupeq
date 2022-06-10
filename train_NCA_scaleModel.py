from re import I
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.png_Dataset import png_Dataset
from IPython.display import clear_output
from lib.CAModel import CAModel
from lib.CAModel_scale import CAModel_scale
from lib.CAModel_Noise import CAModel_Noise
from lib.CAModel_deeper import CAModel_Deeper
from lib.CAModel_Residual import CAModel_Residual
from lib.CAModel_partiallyDead import CAModel_partiallyDead
from lib.CAModel_optimizedTraining import CAModel_optimizedTraining
from lib.CAModel_learntPerceive import CAModel_learntPerceive
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from src.losses.LossFunctions import DiceLoss, DiceBCELoss, DiceLossV2
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_NCA_optTrain import Agent_OptTrain
from src.agents.Agent_NCA import Agent
from src.agents.Agent_NCA_scaleModel import Agent_ScaleSize
import sys
import os
import torchmetrics as tm

# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config = [{
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train_tiny\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train_tiny\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'M:\MasterThesis\Git\NCA\models\NCA_Test41_dataloader_c16_l16e4_scaleChannels_shortInf',
    'device':"cuda:0",
    'n_epoch': 800,
    # Learning rate
    'lr': 16e-4, #16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [64],
    'inference_steps_test': 64,
    # Training config
    'save_interval': 10,
    'evaluate_interval': 10,
    # Model config
    'channel_n': 64,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 10,
    'repeat_factor': 1,
    'input_channels': 3,
    'input_fixed': True,
    'output_channels': 3,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.9,
    'save_pool': False,
    'Persistence': False,
}#,
#{
#    'n_epoch': 2000,
#    'Persistence': True,
#}
]

# Define Experiment
dataset = Nii_Gz_Dataset()
device = torch.device(config[0]['device'])
ca = CAModel_scale(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=8).to(device)
agent = Agent_ScaleSize(ca) #_ScaleSize
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

#loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLossV2()
loss_function = tm.functional.dice

print(loss_function.is_differentiable())

agent.train(data_loader, loss_function)

#exp.temporarly_overwrite_config(config)
#agent.getAverageDiceScore()
#agent.test(data_loader, loss_function)


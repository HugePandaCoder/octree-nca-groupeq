from ctypes import resize
from re import I
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
from IPython.display import clear_output
from lib.CAModel_Noise import CAModel_Noise
from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from src.losses.LossFunctions import DiceLoss, DiceBCELoss
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_NCA import Agent
from src.utils.collate_variable_size import collate_variable_size
import sys
import os

# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config = [{
        'out_path': r"D:\PhD\NCA_Experiments",
        'img_path': r"M:\MasterThesis\Datasets\Hippocampus\original_dataset\imagesTr",
        'label_path': r"M:\MasterThesis\Datasets\Hippocampus\original_dataset\labelsTr",
        'data_type': '.nii.gz', # .nii.gz, .jpg
        'model_path': r'models/NCA_Test29_dataloader_3D_c64_l16e4_beternoise_full',
        'device':"cuda:0",
        'n_epoch': 400,
        # Learning rate
        'lr': 16e-4,
        'lr_gamma': 0.9999,
        'betas': (0.5, 0.5),
        'inference_steps': [64],
        # Training config
        'save_interval': 10,
        'evaluate_interval': 10,
        # Model config
        'channel_n': 64,        # Number of CA state channels
        'target_padding': 0,    # Number of pixels used to pad the target image border
        'target_size': 64,
        'cell_fire_rate': 0.5,
        'cell_fire_interval':None,
        'batch_size': 8,
        'repeat_factor': 1,
        'input_channels': 3,
        'input_fixed': True,
        'output_channels': 3,
        # Data
        'input_size': (64, 64),
        'data_split': [0.6, 0, 0.4], 
        'pool_chance': 0.5,
        'Persistence': False,
    }#,
    #{
    #    'n_epoch': 2000,
    #    'Persistence': True,
    #}
    ]
    # Define Experiment
    dataset = Dataset_NiiGz_3D(slice=2)
    device = torch.device(config[0]['device'])
    ca = CAModel_Noise(config[0]['channel_n'], config[0]['cell_fire_rate'], device).to(device)
    agent = Agent(ca)
    exp = Experiment(config, dataset, ca, agent)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size')) #, collate_fn=collate_variable_size

    loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
    #loss_function = F.mse_loss
    #loss_function = DiceLoss()
    agent.train(data_loader, loss_function)

if __name__ == '__main__':
    main()
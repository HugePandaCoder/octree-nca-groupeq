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
from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from src.losses.LossFunctions import DiceLoss, DiceBCELoss, BCELoss
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_NCA import Agent
from src.utils.collate_variable_size import collate_variable_size
from src.agents.Agent_NCA_optTrain import Agent_OptTrain
from lib.CAModel_optimizedTraining import CAModel_optimizedTraining
import sys
import os

# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config = [{
        'out_path': r"D:\PhD\NCA_Experiments",
        'img_path': r"M:\\MasterThesis\\Datasets\\Prostate\\original_dataset\\ISBI\\Images",
        'label_path': r"M:\\MasterThesis\\Datasets\\Prostate\\original_dataset\\ISBI\\Labels",
        'data_type': '.nii.gz', # .nii.gz, .jpg
        'model_path': r'models/NCA_Test34_dataloader_3D_c64_l16e4_prostate_full_opt_change3',
        'device':"cuda:0",
        'n_epoch': 200,
        # Learning rate
        'lr': 16e-4,
        'lr_gamma': 0.9999,
        'betas': (0.5, 0.5),
        'inference_steps': [64],
        # Training config
        'save_interval': 2,
        'evaluate_interval': 2,
        # Model config
        'channel_n': 16,        # Number of CA state channels
        'target_padding': 0,    # Number of pixels used to pad the target image border
        'target_size': 64,
        'cell_fire_rate': 0.5,
        'cell_fire_interval':None,
        'batch_size': 16,
        'repeat_factor': 1,
        'input_channels': 3,
        'input_fixed': True,
        'output_channels': 3,
        # Data
        'input_size': (256, 256),
        'data_split': [0.8, 0, 0.2], 
        'pool_chance': 0.5,
        'Persistence': False,
    }#,
    #{
    #    'n_epoch': 2000,
    #    'Persistence': True,
    #}
    ]
    # Define Experiment
    dataset = Dataset_NiiGz_3D(slice=2, resize=True)
    device = torch.device(config[0]['device'])
    ca = CAModel_optimizedTraining(config[0]['channel_n'], config[0]['cell_fire_rate'], device).to(device)
    agent = Agent_OptTrain(ca)
    exp = Experiment(config, dataset, ca, agent)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size')) #, collate_fn=collate_variable_size

    loss_function = BCELoss()
    #loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
    #loss_function = F.mse_loss
    #loss_function = DiceLoss()
    agent.train(data_loader, loss_function)

if __name__ == '__main__':
    main()
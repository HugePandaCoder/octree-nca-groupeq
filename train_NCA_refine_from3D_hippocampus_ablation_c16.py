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
from src.datasets.Nii_Gz_Dataset_3D_refine import Dataset_NiiGz_3D_refine
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
from IPython.display import clear_output
#from lib.CAModel import CAModel
#from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks
from src.losses.LossFunctions import DiceLoss, DiceBCELoss
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_NCA_refineResultsUpsample import Agent_RefineResults_Upsample
from src.models.Model_BasicNCA import BasicNCA
from src.models.Model_LearntPerceiveNCA import LearntPerceiveNCA
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
        'img_path': r"/home/jkalkhof_locale/Documents/Data/Hippocampus/preprocessed_dataset_train/imagesTr/",
        'label_path': r"/home/jkalkhof_locale/Documents/Data/Hippocampus/preprocessed_dataset_train/labelsTr/",
        #'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full_Combined_Test/imagesTs/",
        #'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full_Combined_Test/labelsTs/",
        'data_type': '.nii.gz', # .nii.gz, .jpg
        'model_path': r'/home/jkalkhof_locale/Documents/Models/TestNCA_hippocampusFull_Slices_refine_Ablation_C16_v2',
        'device':"cuda:0",
        'n_epoch': 1350,
        # Learning rate
        'lr': 16e-4,
        'lr_gamma': 0.9999,
        'betas': (0.5, 0.5),
        'inference_steps': [32],
        # Training config
        'save_interval': 10,
        'evaluate_interval': 10,
        # Model config
        'channel_n': 16,        # Number of CA state channels
        'target_padding': 0,    # Number of pixels used to pad the target image border
        'target_size': 64,
        'cell_fire_rate': 0,
        'cell_fire_interval':None,
        'batch_size': 640,
        'repeat_factor': 1,
        'input_channels': 3,
        'input_fixed': True,
        'output_channels': 3,
        'stacked_models': 2,
        'scaling_factor': 4, # each axis
        'train_model': 1,
        # Data
        'input_size': [(16, 16), (64, 64)],
        #'data_split': [0.7, 0, 0.3], 
        'data_split': [0.7, 0, 0.3], 
        'pool_chance': 0.5,
        'Persistence': False,
        'unlock_CPU': True,
    }
    ]
    # Define Experiment
    dataset = Nii_Gz_Dataset() #Dataset_NiiGz_3D(slice=2)
    #dataset = Dataset_NiiGz_3D(slice=2)
    device = torch.device(config[0]['device'])

    # Define all model levels
    ca_lvl0 = LearntPerceiveNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=128).to(device)
    ca_lvl1 = LearntPerceiveNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=128).to(device)
    ca = [ca_lvl0, ca_lvl1]

    agent = Agent_RefineResults_Upsample(ca)
    exp = Experiment(config, dataset, ca, agent)
    exp.set_model_state('train')
    dataset.set_experiment(exp)
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size')) #, collate_fn=collate_variable_size

    loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
    #loss_function = F.mse_loss
    #loss_function = DiceLoss()
    
    #agent.train(data_loader, loss_function)
    print(sum(p.numel() for p in ca_lvl0.parameters() if p.requires_grad))
    #agent.getAverageDiceScore()
    #exit()

    exp.temporarly_overwrite_config(config)
    
    print(sum(p.numel() for p in ca_lvl0.parameters() if p.requires_grad))
    
    loss_log = agent.getAverageDiceScore()


    exit()
    with open(r"/home/jkalkhof_locale/Documents/temp/OutTxt/test.txt", "a") as myfile:
        log = {}        
        for x in range(0, 7, 1): #388
            #config[0]['input_size'] = [(256/4, x/4), (256, x)]
            #config[0]['input_size'] = [(x/4, 256/4), (x, 256)]
            config[0]['anisotropy'] = x  
            exp.temporarly_overwrite_config(config)
            loss_log = agent.getAverageDiceScore()
            log[x] = loss_log[0]
        myfile.write(str(log))
            #return sum(loss_log.values())/len(loss_log)

if __name__ == '__main__':
    main()
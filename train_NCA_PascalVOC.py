from re import I
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.Model_BasicNCA import BasicNCA
from lib.CAModel_deeper import CAModel_Deeper
from lib.CAModel_learntPerceive import CAModel_learntPerceive
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass
from src.datasets.Nii_Gz_Dataset_allpass import Nii_Gz_Dataset_allPass
from src.datasets.PascalVOC_Dataset import PascalVOC_Dataset
from src.datasets.png_Dataset import png_Dataset
from IPython.display import clear_output
from src.models.Model_BasicNCA import BasicNCA
from src.models.Model_BasicNCA_noAddUp import BasicNCA_noAddUp
from src.losses.LossFunctions import DiceLoss, DiceBCELoss
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_NCA import Agent
import sys
import os

# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config = [{
    'out_path': r"D:/PhD/NCA_Experiments",
    'img_path': r"M:/MasterThesis/Datasets/VOC2012/Small/JPEGImages_min/",
    'label_path': r"M:/MasterThesis/Datasets/VOC2012/Small/SegmentationClass/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'M:/Models/TestNCA_normal_PascalVOC4',
    'device':"cuda:0",
    'n_epoch': 200,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [64],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 10,
    'ood_interval':100,
    # Model config
    'channel_n': 64,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 6,
    'repeat_factor': 1,
    'input_channels': 3,
    'input_fixed': True,
    'output_channels': 24,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.5,
    'Persistence': False,
    'unlock_CPU': True,
}#,
#{
#    'n_epoch': 2000,
#    'Persistence': True,
#}
]

# Define Experiment
dataset = PascalVOC_Dataset()
device = torch.device(config[0]['device'])
ca = CAModel_learntPerceive(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=128).to(device)
agent = Agent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()
#
with torch.autograd.set_detect_anomaly(True):
    agent.train(data_loader, loss_function)

#exp.temporarly_overwrite_config(config)
#agent.ood_evaluation(epoch=exp.currentStep)
#agent.getAverageDiceScore()
#agent.test(data_loader, loss_function)

from re import I
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
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_distanceField import Nii_Gz_Dataset_DistanceField
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass
from src.datasets.Nii_Gz_Dataset_allpass import Nii_Gz_Dataset_allPass
from src.datasets.png_Dataset import png_Dataset
from IPython.display import clear_output
from src.models.Model_BasicNCA import BasicNCA
from src.models.Model_BasicNCA_noAddUp import BasicNCA_noAddUp
from src.models.Model_LearntPerceiveNCA import LearntPerceiveNCA
from src.models.Model_LearntPerceiveNCA import LearntPerceiveNCA
from src.losses.LossFunctions import DiceLoss, DiceBCELoss, DiceFocalLoss
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_NCA import Agent

from src.agents.Agent_NCA_3dOptVRAM import Agent_NCA_3dOptVRAM

import sys
import os
#from medcam import medcam 
# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config = [{
    'out_path': r"D:/PhD/NCA_Experiments",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'/home/jkalkhof_locale/Documents/Models/NCA3d_optVRAM_prostate_Test3_SingleBatch',
    'device':"cuda:0",
    'n_epoch': 25000,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [35, 35, 35],
    # Training config
    'save_interval': 25,
    'evaluate_interval': 25,
    'ood_interval':100,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 2,
    'repeat_factor': 1,
    'input_channels': 1,
    'input_fixed': True,
    'output_channels': 1,
    # Data
    'input_size': [(100, 100, 16), (200, 200, 32), (400, 400, 64)] ,
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.5,
    'Persistence': False,
    'unlock_CPU': True,
    'train_model':2,
}#,
#{
#    'n_epoch': 2000,
#    'Persistence': True,
#}
]

# Define Experiment
dataset = Dataset_NiiGz_3D()#_lowPass(filter="random")
device = torch.device(config[0]['device'])
ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
ca3 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
ca4 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
ca5 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
ca =[ca1, ca2, ca3] 
#ca = medcam.inject(ca, output_dir=r"M:\AttentionMapsUnet", save_maps = True)
agent = Agent_NCA_3dOptVRAM(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()
#

#with torch.autograd.set_detect_anomaly(True):
agent.train(data_loader, loss_function)

#exp.temporarly_overwrite_config(config)

#agent.getAverageDiceScore()

#agent.ood_evaluation(epoch=exp.currentStep)
#agent.test(data_loader, loss_function)

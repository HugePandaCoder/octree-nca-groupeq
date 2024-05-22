from re import I
from re import I
import time
import numpy as np
import torch
from src.models.Model_BasicNCA import BasicNCA
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from IPython.display import clear_output
from src.models.Model_BasicNCA import BasicNCA
from src.losses.LossFunctions import DiceLoss, DiceBCELoss, DiceFocalLoss
from src.utils.Experiment import Experiment, DataSplit
from src.agents.Agent_M3D_NCA import Agent_M3D_NCA
import sys
import os
#from medcam import medcam
# TODO REMOVE!!!
#import warnings
#warnings.filterwarnings("ignore")
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
seed_value = 87465874
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
config = [{
    'name': r'train_M3D_NCA_prostate2_2',
    'out_path': r"/local/scratch/clmn1/octree_study/",
    'img_path': r"/local/scratch/jkalkhof/Data/Prostate_MEDSeg/imagesTr/",
    'label_path': r"/local/scratch/jkalkhof/Data/Prostate_MEDSeg/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'/local/scratch/clmn1/octree_study/Models/NCA3d__prostate5',
    'device':"cuda:0",
    'n_epoch': 2000,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'inference_steps': [20, 40],
    # Training config
    'save_interval': 25,
    'evaluate_interval': 100,
    'ood_interval':100,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 3,
    'batch_duplication': 2,
    'input_channels': 1,
    'input_fixed': True,
    'output_channels': 1,
    # Data
    'input_size': [(80, 80, 6), (320, 320, 24)] ,
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.3],
    'pool_chance': 0.5,
    'keep_original_scale': True,
    'rescale': True,
    'Persistence': False,
    'unlock_CPU': True,
    'train_model':1,
    'hidden_size':64,
    #'bad_samples':0.3,
}#,
#{
#    'n_epoch': 2000,
#    'Persistence': True,
#}
]
# Define Experiment
dataset = Dataset_NiiGz_3D()#_lowPass(filter="random")
device = torch.device(config[0]['device'])
ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7).to(device)
ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3).to(device)
#ca3 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
#ca4 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
#ca5 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64).to(device)
ca =[ca1, ca2]
#ca = medcam.inject(ca, output_dir=r"M:\AttentionMapsUnet", save_maps = True)
agent = Agent_M3D_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceFocalLoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()
#
#with torch.autograd.set_detect_anomaly(True):
print("MODEL # PARAMETERS", sum((sum(p.numel() for p in m.parameters() if p.requires_grad)) for m in ca))
agent.train(data_loader, loss_function)
#exp.temporarly_overwrite_config(config)
agent.getAverageDiceScore(pseudo_ensemble=True)
#agent.ood_evaluation(epoch=exp.currentStep)
#agent.test(data_loader, loss_function)
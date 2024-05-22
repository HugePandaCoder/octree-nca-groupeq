
# %%
import os
import sys

script_path = os.path.abspath(__file__)  # Gets the absolute path of the current file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path))) # Adjust the path to the project root
print(project_root)
sys.path.append(project_root)


import time

import torch

from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import \
    Dataset_NiiGz_3D_customPath
from src.losses.LossFunctions import DiceFocalLoss
from src.models.Model_M3DNCA import M3DNCA
from src.models.Model_M3DNCA_alive import M3DNCA_alive
from src.utils.Experiment import Experiment

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from src.utils.ProjectConfiguration import ProjectConfiguration
ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"

config = [{
    'img_path': r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/imagesTr/",
    'label_path': r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/labelsTr/",
    'name': r'MIA_prostate_M3D_NCA_Run6', #12 or 13, 54 opt, 
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 50,
    'evaluate_interval': 1501,
    'n_epoch': 1500,
    'batch_duplication': 2,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [20, 40],
    'cell_fire_rate': 0.5,
    'batch_size': 3,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(80, 80, 6), (320, 320, 24)] ,
    'scale_factor': 4,
    'data_split': [1.0, 0, 0.0], 
    'keep_original_scale': False,
    'rescale': True,
}
]

dataset = Dataset_NiiGz_3D(store=True)
device = torch.device(config[0]['device'])
ca1 = M3DNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels'], output_channels=config[0]['output_channels'], levels=2, scale_factor=config[0]['scale_factor'], steps=config[0]['inference_steps']).to(device)
ca = ca1
agent = M3DNCAAgent(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
agent.train(data_loader, loss_function)

if True:

    hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(320, 320, 24), imagePath=r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/imagesTs", labelPath=r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/labelsTs")
    hyp99_test.exp = exp


    start_time = time.perf_counter()
    agent.getAverageDiceScore(pseudo_ensemble=True, dataset=hyp99_test)

    end_time = time.perf_counter()


    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to execute.")

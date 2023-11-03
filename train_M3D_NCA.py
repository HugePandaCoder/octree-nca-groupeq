# %% [markdown]
# # M3D-NCA: Robust 3D Segmentation with Built-in Quality Control
# ### John Kalkhof, Anirban Mukhopadhyay
# __https://arxiv.org/pdf/2309.02954.pdf__
# 
# 
# 
# ***

# %% [markdown]
# ## __The Backbone Model__
# <div>
# <img src="src/images/model_M3DNCA.png" width="600"/>
# </div>

# %% [markdown]
# ## _1. Imports_

# %%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3D_NCA import Agent_M3D_NCA

config = [{
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
    'name': r'M3D_NCA_Run15',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 25,
    'evaluate_interval': 25,
    'n_epoch': 3000,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [20, 20, 20, 20],
    'cell_fire_rate': 0.5,
    'batch_size': 4,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':3,
    # Data
    'input_size': [(40, 40, 4), (80, 80, 8), (160, 160, 16), (320, 320, 32)] ,
    'scale_factor': 2,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': False,
    'rescale': True,
}
]

dataset = Dataset_NiiGz_3D()
device = torch.device(config[0]['device'])
ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
ca3 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
ca4 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
ca = [ca1, ca2, ca3, ca4]
agent = Agent_M3D_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() 

#agent.train(data_loader, loss_function)

agent.getAverageDiceScore(pseudo_ensemble=False)



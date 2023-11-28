#%%
import torch
from src.datasets.Nii_Gz_Dataset_3D_gen import Dataset_NiiGz_3D_gen
from src.models.Model_GenNCA import GenNCA
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_NCA_gen import Agent_NCA_gen

from src.models.Model_BasicNCA3D import BasicNCA3D
from src.agents.Agent_NCA import Agent_NCA

config = [{
    # Basic
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    'name': r"genMRIseg_33",#_baseline", 75% with vec, 77.5% baseline
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 5,#
    'evaluate_interval': 5,
    'n_epoch': 1000,
    'batch_size': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': 10,
    'cell_fire_rate': 0.5,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    # Data
    'input_size': (32, 32, 26),
    'data_split': [0.7, 0, 0.3], 
}
]

dataset = Dataset_NiiGz_3D_gen(extra_channels=2)
device = torch.device(config[0]['device'])

ca = GenNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], extra_channels=2).to(device)
agent = Agent_NCA_gen(ca)

#ca = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels']).to(device)
#agent = Agent_NCA_gen(ca)

exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceFocalLoss() 

agent.train(data_loader, loss_function)

agent.getAverageDiceScore()



# %%

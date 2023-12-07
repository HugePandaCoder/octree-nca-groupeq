#%%
import torch
from src.datasets.png_Dataset_gen import png_Dataset_gen
from src.models.Model_GenNCA import GenNCA
from src.models.Model_GenNCA_v3 import GenNCA_v3
from src.utils.Experiment import Experiment
from src.agents.Agent_NCA_genImage import Agent_NCA_genImage

from src.models.Model_BasicNCA3D import BasicNCA3D
from src.agents.Agent_NCA import Agent_NCA
import os
import torch.nn as nn

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


config = [{
    # Basic
    'img_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/",
    'name': r"headlessImageGen_53_celebA",#_baseline", 75% with vec, 77.5% baseline
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 3e-5,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 50,#
    'evaluate_interval': 50,
    'n_epoch': 5000,
    'batch_size': 8,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': 30,
    'cell_fire_rate': 0.5,
    'input_channels': 3,
    'output_channels': 3,
    'hidden_size': 64,
    'extra_channels': 6,
    # Data
    'input_size': (22, 22),
    'data_split': [0.001, 0.9989, 0.0001],
}
]
dataset = png_Dataset_gen(extra_channels=config[0]['extra_channels'])
device = torch.device(config[0]['device'])

ca = GenNCA_v3(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], extra_channels=config[0]['extra_channels'], kernel_size=3, batch_size=config[0]['batch_size']).to(device)
agent = Agent_NCA_genImage(ca)

#ca = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels']).to(device)
#agent = Agent_NCA_gen(ca)

exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = nn.MSELoss()

agent.train(data_loader, loss_function)

agent.getAverageDiceScore()



# %%

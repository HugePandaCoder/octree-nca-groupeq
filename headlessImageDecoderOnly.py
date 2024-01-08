#%%
import torch
from src.datasets.png_Dataset_gen import png_Dataset_gen
from src.datasets.png_Dataset import png_Dataset
from src.models.Model_GenNCA import GenNCA
from src.models.Model_GenNCA_v3 import GenNCA_v3
from src.models.Model_GenNCA_v3_noHyper import GenNCA_V3_NoHyper
from src.models.Model_GrowingNCA import GrowingNCA   
from src.models.DecoderOnly_Simple import DecoderOnly_Simple
from src.models.Autoencoder import Autoencoder
from src.utils.Experiment import Experiment
from src.agents.Agent_NCA_genDecoder import Agent_NCA_gen_Decoder
from src.agents.Agent_Growing import Agent_Growing

from src.models.Model_BasicNCA3D import BasicNCA3D
from src.agents.Agent_NCA import Agent_NCA
import os
import torch.nn as nn
import torch.nn.functional as F

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


config = [{
    # Basic
    'img_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/",
    'name': r"headlessImageGen_269_celebA_DecoderOnly",#_baseline", 75% with vec, 77.5% baseline
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 3e-4,
    'lr_gamma': 0.99999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 20,#
    'evaluate_interval': 2,
    'n_epoch': 50000,
    'batch_size': 24,
    # Model
    'channel_n': 48,        # Number of CA state channels
    'inference_steps': 30,
    'cell_fire_rate': 0.1,
    'input_channels': 4,
    'output_channels': 3,
    'hidden_size': 512,
    'extra_channels': 32,
    # Data
    'input_size': (128, 128),
    'data_split': [0.064, 0.9359, 0.0001],
    #'data_split': [0.0005, 0.9994, 0.0001],
}
]

#dataset = png_Dataset(setSeed=True, buffer=True) #
#dataset.set_normalize(True)
dataset = png_Dataset_gen(extra_channels=config[0]['extra_channels'])
device = torch.device(config[0]['device'])


#ca = GenNCA_v3(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], extra_channels=config[0]['extra_channels'], kernel_size=3, batch_size=config[0]['batch_size']).to(device)
ca = DecoderOnly_Simple(config[0]['batch_size'], config[0]['extra_channels'], device, hidden_size=config[0]['hidden_size']).to(device)
#ca = GenNCA_V3_NoHyper(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], extra_channels=config[0]['extra_channels'], kernel_size=3, batch_size=config[0]['batch_size']).to(device)
#ca = GrowingNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size']).to(device)

print("PARAMETERS", sum(p.numel() for p in ca.parameters() if p.requires_grad))

agent = Agent_NCA_gen_Decoder(ca)
#agent = Agent_Growing(ca)

#ca = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels']).to(device)
#agent = Agent_NCA_gen(ca)

exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = F.mse_loss#nn.MSELoss()          

agent.train(data_loader, loss_function)

agent.getAverageDiceScore()



# %%

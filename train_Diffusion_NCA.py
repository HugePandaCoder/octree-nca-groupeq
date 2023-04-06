#%%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
from src.models.Model_DiffusionNCA import DiffusionNCA
from src.models.Model_DiffusionNCA_fft import DiffusionNCA_fft
from src.models.Model_DiffusionNCA_fft2 import DiffusionNCA_fft2
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_Diffusion import Agent_Diffusion

config = [{
    # Basic
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Emojis_Smiley/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Emojis_Smiley/", #img_align_celeba
    'name': r'DiffusionNCA_Run131_fft_Emojis_Smiley_alivePulse_fixed',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 100,
    'evaluate_interval': 100,
    'n_epoch': 100000,
    'batch_size': 36,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'batch_duplication': 1,
    'inference_steps': 64,
    'cell_fire_rate': 0.5,
    'input_channels': 3,
    'output_channels': 3,
    'hidden_size': 128,
    # Data
    'input_size': (36, 36),
    'data_split': [0.9, 0, 0.1], 
    'timesteps': 200,
    '2D': True,
}
]

#dataset = Dataset_NiiGz_3D(slice=2)
dataset= png_Dataset()
device = torch.device(config[0]['device'])
ca = DiffusionNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
agent = Agent_Diffusion(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() 

agent.train(data_loader, loss_function)

#agent.getAverageDiceScore()


# %%

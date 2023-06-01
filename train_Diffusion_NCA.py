#%%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
from src.models.Model_DiffusionNCA import DiffusionNCA
from src.models.Model_DiffusionNCA_Group import DiffusionNCA_Group
from src.models.Model_DiffusionNCA_fft import DiffusionNCA_fft
from src.models.Model_DiffusionNCA_fft2 import DiffusionNCA_fft2
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_Diffusion import Agent_Diffusion

config = [{
    # Basic
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba/", #img_align_celeba, Emojis_Smiley, Emojis_Google
    'name': r'DiffusionNCA_Run314_CelebA_fixed_rescale_norm_fft_updat_l1_k7_grouping_newModel_wAlive',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 160e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 25,
    'evaluate_interval': 25,
    'n_epoch': 100000,
    'batch_size': 8,
    # Model
    'channel_n': 32,        # Number of CA state channels
    'batch_duplication': 1,
    'inference_steps': 20,
    'cell_fire_rate': 0.5,
    'input_channels': 3,
    'output_channels': 3,
    'hidden_size': 2048,
    # Data
    'input_size': (48, 48),
    'data_split': [0.005, 0, 1], 
    'timesteps': 500,
    '2D': True,
}
]

#dataset = Dataset_NiiGz_3D(slice=2)
dataset = png_Dataset()
device = torch.device(config[0]['device'])
#ca = DiffusionNCA_Group(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
ca = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
agent = Agent_Diffusion(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() 

#agent.train(data_loader, loss_function)

agent.generateSamples(samples=1)


# %%

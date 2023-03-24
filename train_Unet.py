from unet import UNet2D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_UNet import Agent

config = [{
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full_Slices/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full_Slices/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'/home/jkalkhof_locale/Documents/Models/UNet_Test8_Scaled',
    'device':"cuda:0",
    'n_epoch': 1000,
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'inference_steps': [64],
    # Training config
    'save_interval': 100,
    'evaluate_interval': 10,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 1,
    'persistence_chance':0.5,
    # Data
    'input_size': (256, 256),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.7,
    'Persistence': False,
    'output_channels': 3,
}]

# Define Experiment
dataset = Nii_Gz_Dataset()
device = torch.device(config[0]['device'])
ca = UNet2D(in_channels=3, padding=1, out_classes=3).to(device)
agent = Agent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceBCELoss() 

# Number of parameters
print(sum(p.numel() for p in ca.parameters() if p.requires_grad))

# Train Model
agent.train(data_loader, loss_function)

# Average Dice Score on Test set
agent.getAverageDiceScore()
from unet import UNet2D
from src.models.Model_MedNCA import MedNCA
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_UNet import Agent
from src.agents.Agent_MedNCA_Simple import MedNCAAgent

config = [{
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    'name': r'MedNCA_Run3',
    'device':"cuda:0",
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 2,
    'evaluate_interval': 2,
    'n_epoch': 1000,
    'batch_size': 40,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'cell_fire_rate': 0.5,
    'output_channels': 3,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 

}]

# Define Experiment
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca = MedNCA(channel_n=17, fire_rate=0.5, steps=64, device = "cuda:0", hidden_size=128, input_channels=1, output_channels=1).to("cuda:0")
agent = MedNCAAgent(ca)
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


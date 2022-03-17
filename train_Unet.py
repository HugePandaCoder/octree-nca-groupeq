# REMOVE: if you use this code for your research please cite: https://zenodo.org/record/3522306#.YhyO1-jMK70
from unet import UNet2D
from src.Datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from Experiment import Experiment, DataSplit
import torch
from src.LossFunctions.LossFunctions import DiceLoss, DiceBCELoss
from src.Agents.Agent_UNet import Agent

import warnings
warnings.filterwarnings("ignore")

config = {
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': "models/unet_test3.pth",
    'reload': True,
    'device':"cuda:0",
    'n_epoch': 40,
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [64],
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 64,
    'persistence_chance':0.5,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.7,
    'Persistence': True,
}

# Define Experiment
dataset = Nii_Gz_Dataset()
device = torch.device(config[0]['device'])
ca = UNet2D(in_channels=3, padding=1, out_classes=3).to(device)
exp = Experiment(config, dataset, ca)
exp.set_model_state('train')

data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])

loss_function = DiceBCELoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()

agent = Agent(ca, exp)
agent.train(dataset, loss_function)

exit()

# Define Experiment
dataset = Nii_Gz_Dataset(config['input_size'])
exp = Experiment(config, dataset)
exp.set_model_state('train')

device = torch.device(config['device'])

model = UNet2D(in_channels=3, padding=1, out_classes=3).to(device)

if config['reload'] == True:
    print("Model loaded")
    model.load_state_dict(torch.load(config['model_path']))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'])
agent = Agent(model, config)

loss_function = DiceBCELoss()

agent.train(data_loader, 40, loss_function)




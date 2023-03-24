from re import I
from re import I
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.agents.Agent_NCA import Agent_NCA

config = [{
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'/home/jkalkhof_locale/Documents/Models/NCA3d_refactored_test6',
    'device':"cuda:0",
    'n_epoch': 30000,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': 20,
    # Training config
    'save_interval': 10,
    'evaluate_interval': 1,
    # Model config
    'channel_n': 8,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 4,
    'repeat_factor': 1,
    'input_channels': 1,
    'input_fixed': True,
    'output_channels': 1,
    # Data
    'input_size': (64, 64, 52),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.5,
    'Persistence': False,
    'unlock_CPU': True,
    'keep_original_scale': True,
    'rescale': True,
}
]

# Define Experiment
dataset = Dataset_NiiGz_3D()
device = torch.device(config[0]['device'])
ca = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=64, input_channels=config[0]['input_channels']).to(device)
agent = Agent_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceFocalLoss() 

# Run training
agent.train(data_loader, loss_function)

# Average Dice Score on Test set
#agent.getAverageDiceScore()

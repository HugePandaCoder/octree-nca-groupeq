from unet import UNet2D
from src.models.Model_MedNCA import MedNCA
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.agents.Agent_UNet import UNetAgent

config = [{
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Dataset_BUSI_with_GT/image/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Dataset_BUSI_with_GT/label/",
    'name': r'MedNCA_Run13_Breast',
    'device':"cuda:0",
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 2,
    'evaluate_interval': 2,
    'n_epoch': 1000,
    'batch_size': 12,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'cell_fire_rate': 0.5,
    'input_channels': 1,
    'output_channels': 1,
    # Data
    'input_size': (320, 320),
    'data_split': [0.1, 0.89, 0.01], 

}]

from src.datasets.png_seg_Dataset import png_seg_Dataset
dataset = png_seg_Dataset(buffer=True)


# Define Experiment
#dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca = MedNCA(channel_n=16, fire_rate=0.1, steps=32, device = "cuda:0", hidden_size=128, input_channels=1, output_channels=1).to("cuda:0")
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


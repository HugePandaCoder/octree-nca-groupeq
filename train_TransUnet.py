
# %%
from unet import UNet3D
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_UNet import Agent
import time
#from transunet import TransUNet
from src.models.Model_TransUnet import TransUNet

config = [{
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
    'name': r'TransUnet2D_Run7',
    'device':"cuda:0",
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 200,
    'evaluate_interval': 25,
    'n_epoch': 1000,
    'batch_size': 16,
    # Data
    'input_size': (320, 320),
    'data_split': [0.7, 0, 0.3], 

}]

# Define Experiment
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca = TransUNet(img_dim=320, in_channels=1, out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=1).to(device) #in_channels=1, padding=1, out_classes=1

# Load TransUNet weights

agent = Agent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceBCELoss() 

# Number of parameters
print("Nr. Params.: ", sum(p.numel() for p in ca.parameters() if p.requires_grad))

agent.train(data_loader, loss_function)


start_time = time.perf_counter()
#agent.getAverageDiceScore(pseudo_ensemble=False)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")



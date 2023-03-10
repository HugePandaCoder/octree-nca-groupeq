import torch
from src.datasets.Nii_Gz_Dataset_3D_zeros import Dataset_NiiGz_3D_zeros
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Experiment import Experiment
from src.models.Model_BasicNCA3D_Big import BasicNCA3D_Big

from src.agents.Agent_NCA_3dOptVRAM_EfficientInference import Agent_NCA_3dOptVRAM_EfficientInference
import datetime


import sys
import os
#from medcam import medcam 
# TODO REMOVE!!! 
#import warnings
#warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config = [{
    'out_path': r"D:/PhD/NCA_Experiments",
    'img_path': r"/home/pi/Data/Empty/imagesTr/", #r"M:/data/Irgendwas/imagesTr", #
    'label_path': r"/home/pi/Data/Empty/labelsTr/", #r"M:/data/Irgendwas/labelsTr", #
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'/home/pi/Models/setup2_v1',
    'device':"cpu",
    'n_epoch': 25000,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'inference_steps': [30],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 100,
    'ood_interval':100,
    # Model config
    'channel_n': 16,        # Number of CA state channels
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
    'input_size': [(64, 64, 52)] ,
    'scale_factor': 4,
    'data_split': [0.5, 0, 0.5], 
    'pool_chance': 0.5,
    'keep_original_scale': True,
    'rescale': True,
    'Persistence': False,
    'unlock_CPU': True,
    'train_model':0,
    'hidden_size':64,
}#,
#{
#    'n_epoch': 2000,
#    'Persistence': True,
#}
]

start = datetime.datetime.now()
print("START_TIME: ", start)
# Define Experiment
dataset = Dataset_NiiGz_3D_zeros()#_lowPass(filter="random")
device = torch.device(config[0]['device'])
ca1 = BasicNCA3D_Big(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7).to(device)
#ca2 = BasicNCA3D_Big(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3).to(device)
ca =[ca1]#, ca3, ca4] 
#ca = medcam.inject(ca, output_dir=r"M:\AttentionMapsUnet", save_maps = True)
agent = Agent_NCA_3dOptVRAM_EfficientInference(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()
#

#with torch.autograd.set_detect_anomaly(True):
#agent.train(data_loader, loss_function)

#exp.temporarly_overwrite_config(config)

agent.getAverageDiceScore()
end = datetime.datetime.now()
print("END_TIME: ", end)
print("PASSED_TIME: ", end-start)
#agent.ood_evaluation(epoch=exp.currentStep)
#agent.test(data_loader, loss_function)

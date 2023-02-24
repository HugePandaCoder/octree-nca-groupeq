# REMOVE: if you use this code for your research please cite: https://zenodo.org/record/3522306#.YhyO1-jMK70
from unet import UNet3D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass
from src.utils.Experiment import Experiment, DataSplit
import torch
from src.losses.LossFunctions import DiceLoss, DiceBCELoss, DiceFocalLoss
from src.agents.Agent_UNet import Agent
import segmentation_models_pytorch as smp
#from medcam import medcam

import warnings
warnings.filterwarnings("ignore")

config = [{
    'out_path': r"D:\PhD\NCA_Experiments",
    #'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    #'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full_Combined_Test/imagesTs/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_Full_Combined_Test/labelsTs/",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'/home/jkalkhof_locale/Documents/Models/Models/UNet_Hippocampus_MedSeg_1_VGG11',
    'device':"cuda:0",
    'n_epoch': 1000,
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'inference_steps': [64],
    # Training config
    'save_interval': 250,
    'evaluate_interval': 50,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 100,
    'persistence_chance':0.5,
    # Data
    'input_size': (64, 64) ,
    'keep_original_scale': True,
    'rescale': True,
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.7,
    'Persistence': False,
    'output_channels': 1,
    'input_channels': 1,
}]

# Define Experiment
#dataset = Dataset_NiiGz_3D(slice=2) #_3D(slice=2)
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca = smp.Unet(encoder_name='vgg11', encoder_weights=None, in_channels=1, classes=1).to(device)
#ca = medcam.inject(ca, output_dir=r"M:\AttentionMapsUnet", save_maps = True)

agent = Agent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)

data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceFocalLoss() #nn.CrossEntropyLoss() #
#loss_function = F.mse_loss
#loss_function = DiceLoss()

#exp.temporarly_overwrite_config(config)
#agent.ood_evaluation(epoch=exp.currentStep)
#agent.getAverageDiceScore()
#agent.train(data_loader, loss_function)

print(sum(p.numel() for p in ca.parameters() if p.requires_grad))
#exp.temporarly_overwrite_config(config)
agent.getAverageDiceScore()

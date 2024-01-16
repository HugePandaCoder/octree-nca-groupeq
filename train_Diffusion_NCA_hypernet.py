#%%
import torch
#from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.png_Dataset import png_Dataset
from src.models.Model_DiffusionNCA import DiffusionNCA
from src.models.Model_DiffusionNCA_Group import DiffusionNCA_Group
#from src.models.Model_DiffusionNCA_n_level import DiffusionNCA_fft2
from src.models.Model_DiffusionNCA_fft2_sin_hypernet import DiffusionNCA_fft2_hypernet
from src.models.Model_DiffusionNCA_fft2_sin_hypernet_wavelet import DiffusionNCA_wavelet_hypernet
from src.models.Model_DiffusionNCA_fft2_sin_submission import DiffusionNCA_fft2
#from src.models.Model_DiffusionNCA_multilevel import DiffusionNCA_fft2
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_Diffusion import Agent_Diffusion
from src.agents.Agent_Diffusion_oneChain import Agent_Diffusion_Chain
from src.datasets.Dataset_BCSS import Dataset_BCSS
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

config = [{
    # Basic
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Task04_Hippocampus/train/labelsTr/",
    #/home/jkalkhof_locale/Documents/Data/BCSS/BCSS_train/images/
    'img_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/img_align_celeba_64/", #img_align_celeba, Emojis_Smiley, Emojis_Google, img_align_celeba_64
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/BCSS/BCSS_train/images/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/BCSS/BCSS_train/images/",
    'name': r'DiffusionNCA_Run937_CelebA_hypernet', #last 58 #DiffusionNCA_Run585_CelebA_fixed_rescale_norm_fft_updat_l2_k7_multiNCA_4_smoothl1_twoStep
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4, #32 16e-4
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 5,
    'evaluate_interval': 1,
    'n_epoch': 100000,
    'batch_size': 12,
    # Model
    'channel_n': 64,        # Number of CA state channels
    'batch_duplication': 1,
    'inference_steps': 10,
    'cell_fire_rate': 0.1,
    'input_channels': 3,
    'output_channels': 3,
    'hidden_size':  384,
    'schedule': 'cosine',
    # Data
    'input_size': (64, 64),
    'data_split': [0.005, 0.94, 0.001],#[0.80340968, 0.09806, 1], 
    'timesteps': 300,
    'timesteps_train': 1,
    '2D': True,
    'unlock_CPU': True,
}
]

#dataset = Dataset_NiiGz_3D(slice=2)
dataset = png_Dataset(buffer=True)
device = torch.device(config[0]['device']) 
#ca = DiffusionNCA_Group(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)

#ca1 = DiffusionNCA_fft2_hypernet(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca2 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca3 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca4 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca5 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca6 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca7 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca8 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
#ca0 = DiffusionNCA_fft2(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
ca0 = DiffusionNCA_fft2_hypernet(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels'], img_size=config[0]['input_size'][0],).to(device)
ca = [ca0]#, ca1]#[ca0, ca1]#, ca2, ca3, ca4, ca5, ca6, ca7, ca8, ca9]#, ca2, ca3, ca4, ca5, ca6, ca7, ca8, ca9]

print("PARAMETERS", sum(p.numel() for p in ca0.parameters() if p.requires_grad))
 
#agent = Agent_Diffusion_Chain(ca)
agent = Agent_Diffusion(ca)
exp = Experiment(config, dataset, ca, agent)
#exp.bufferData()
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() 

if True:
    #with torch.autograd.detect_anomaly(): 
    agent.train(data_loader, loss_function)
else:
    #torch.manual_seed(142)
    #agent.calculateFID_fromFiles(samples=100)
    #agent.test_fid(samples=556, optimized=True, saveImg=True)
    agent.generateSamples(samples=1, optimized=True)#, optimized=True)


# %%

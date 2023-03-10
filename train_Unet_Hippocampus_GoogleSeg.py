# REMOVE: if you use this code for your research please cite: https://zenodo.org/record/3522306#.YhyO1-jMK70
from unet import UNet3D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass
from src.utils.Experiment import Experiment, DataSplit
import torch
from src.losses.LossFunctions import DiceLoss, DiceBCELoss, DiceFocalLoss
from src.agents.Agent_NCA import Agent
from src.models.Model_GoogleNCA import GoogleNCA
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
    'model_path': r'/home/jkalkhof_locale/Documents/Models/UNet_Hippocampus_GoogleSeg_1_Scaled',
    'device':"cuda:0",
    'n_epoch': 3000,
    # Learning rate
    'lr': 3e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': 60, #[80, 40],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 50,
    'ood_interval':100,
    # Model config
    'channel_n': 48,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    
    'cell_fire_interval':None,
    'batch_size': 20,
    'repeat_factor': 1,
    'input_channels': 3,
    'input_fixed': True,
    'output_channels': 1,
    # Data
    'input_size': (64, 64), #[(40, 40, 3), (80, 80, 6), (160, 160, 12), (320, 320, 24)] ,
    'scaling_factor': 4,
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.5,
    'keep_original_scale': True,
    'rescale': True,
    'Persistence': False,
    'unlock_CPU': True,
    'train_model':1,
    'hidden_size':64,
}]

# Define Experiment
#dataset = Dataset_NiiGz_3D(slice=2) #_3D(slice=2)
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca = GoogleNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device).to(device)
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
print(sum(p.numel() for p in ca.parameters() if p.requires_grad))
#agent.train(data_loader, loss_function)



#print(sum(p.numel() for p in ca.parameters() if p.requires_grad))
exp.temporarly_overwrite_config(config)
agent.getAverageDiceScore()

exit()
with open(r"/home/jkalkhof_locale/Documents/temp/OutTxt/test.txt", "a") as myfile:
    log = {}        
    for x in range(254, 60, -4): #388
        print(x)
        #config[0]['input_size'] = [(256/4, x/4), (256, x)]
        #config[0]['input_size'] = [(x/4, 256/4), (x, 256)]
        config[0]['anisotropy'] = x  
        exp.temporarly_overwrite_config(config)
        loss_log = agent.getAverageDiceScore()
        log[x] = loss_log[0]
    myfile.write(str(log))
        #return sum(loss_log.values())/len(loss_log)

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




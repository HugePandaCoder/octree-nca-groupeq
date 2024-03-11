from unet import UNet2D
from src.models.Model_MedNCA import MedNCA
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.Experiment import Experiment
import torch
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.agents.Agent_UNet import UNetAgent
from src.datasets.Nii_Gz_Dataset_customPath import Dataset_NiiGz_customPath
from unet import UNet2D
from src.agents.Agent_UNet import UNetAgent 
import segmentation_models_pytorch as smp

config = [{
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/images",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/labels",
    'name': r'MobileNetUNet_Run1_MIMIC_50',
    'device':"cuda:0",
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 500,
    'evaluate_interval': 501,
    'n_epoch': 500,
    'batch_size': 20,
    # Model config
    'channel_n': 32,        # Number of CA state channels
    'cell_fire_rate': 0.3,
    'input_channels': 1,
    'output_channels': 1,
    # Data
    'input_size': (256, 256),
    'data_split': [1.0, 0.0, 0.0], 

}]

#from src.datasets.png_seg_Dataset import png_seg_Dataset
dataset = dataset = Nii_Gz_Dataset()


# Define Experiment
#dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])

#ca = UNet2D(in_channels=1, padding=1, out_classes=1).to(device) #(channel_n=16, fire_rate=0.1, steps=48, device = "cuda:0", hidden_size=128, input_channels=1, output_channels=1).to("cuda:0")
ca = smp.Unet(
    encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
).to(device)

agent = UNetAgent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceBCELoss() 

# Number of parameters
print(sum(p.numel() for p in ca.parameters() if p.requires_grad))


if True:
    # Generate variance and segmentation masks for unseen dataset
    print("--------------- TESTING HYP 99 ---------------")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/images_test", labelPath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/labels_test")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/images_test", labelPath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/labels_test")
    hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/images_test", labelPath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/labels_test")
    hyp99_test.exp = exp
    agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
else:
    # Train Model
    agent.train(data_loader, loss_function)

    # Average Dice Score on Test set
    #agent.getAverageDiceScore()


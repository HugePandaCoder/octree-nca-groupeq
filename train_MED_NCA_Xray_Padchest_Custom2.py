
# %%
import torch
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.models.Model_M3DNCA import M3DNCA
from src.models.Model_MedNCA import MedNCA
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.models.Model_M3DNCA_alive import M3DNCA_alive
from src.losses.LossFunctions import DiceFocalLoss, WeightedDiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.datasets.Nii_Gz_Dataset_customPath import Dataset_NiiGz_customPath
from src.models.Model_MedNCA_finetune import MedNCA_finetune
from src.agents.Agent_Med_NCA_Simple_finetuning import Agent_Med_NCA_finetuning
import time
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

config = [{
    'img_path': r'/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_nii',
    'label_path': r'/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_nii_labels',
    'name': r'Med_NCA_Run8_Padchest_awful_Custom_50', #12 or 13, 54 opt, 
    'pretrained': r'Med_NCA_Run2_Padchest50', #12 or 13, 54 opt, 
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 3e-6,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 500,
    'evaluate_interval': 5001,
    'n_epoch': 500,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [20, 20],
    'cell_fire_rate': 0.5,
    'batch_size': 8,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 128,
    'train_model':1,
    # Data
    'input_size': [(64, 64), (256, 256)] ,
    'scale_factor': 4,
    'data_split': [1.0, 0, 0.0], 
    'keep_original_scale': False,
    'rescale': True,
}
]

dataset = Nii_Gz_Dataset()#store=True)
device = torch.device(config[0]['device'])
ca = MedNCA_finetune(channel_n=16, fire_rate=0.5, steps=50, device = "cuda:0", hidden_size=128, input_channels=1, output_channels=1, batch_duplication=1).to("cuda:0")
agent = Agent_Med_NCA_finetuning(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = WeightedDiceBCELoss() 

if True:
    # Generate variance and segmentation masks for unseen dataset
    print("--------------- TESTING HYP 99 ---------------")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/images_test", labelPath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/MIMIC_50/labels_test")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/images_test", labelPath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/ChestX8_50/labels_test")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/images_test", labelPath=r"/home/jkalkhof_locale/Documents/Data/MICCAI24/Padchest_50/labels_test")
    
    # Custom dataset
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_contrast_nifti", labelPath=r"/home/jkalkhof_locale/Downloads/MICCAI_png_Test/images_preprocessed_nifti_labels")
    hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r'/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_nii', labelPath=r'/home/jkalkhof_locale/Documents/MICCAI24_finetuning/Custom_finetuning/images_preprocessed_awful_nii_labels')
    

    # Generate mean and variance maps
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/MICCAI24_finetuning/MIMIC_50_finetune/MIMIC_50/images", labelPath=r"/home/jkalkhof_locale/Documents/MICCAI24_finetuning/MIMIC_50_finetune/MIMIC_50/labels")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/MICCAI24_finetuning/MIMIC_50_finetune/ChestX8_50/images", labelPath=r"/home/jkalkhof_locale/Documents/MICCAI24_finetuning/MIMIC_50_finetune/ChestX8_50/labels")
    #hyp99_test = Dataset_NiiGz_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/MICCAI24_finetuning/MIMIC_50_finetune/Padchest_50/images", labelPath=r"/home/jkalkhof_locale/Documents/MICCAI24_finetuning/MIMIC_50_finetune/Padchest_50/labels")
        
    
    hyp99_test.exp = exp
    agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)
else:   
    agent.train(data_loader, loss_function)



    #start_time = time.perf_counter()
    #agent.getAverageDiceScore(pseudo_ensemble=False)
    #end_time = time.perf_counter()

    #elapsed_time = end_time - start_time
    #print(f"The function took {elapsed_time} seconds to execute.")

# %%

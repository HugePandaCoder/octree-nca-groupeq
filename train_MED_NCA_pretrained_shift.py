# %%
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_M3DNCA import M3DNCA
from src.models.Model_MedNCA_finetune import MedNCA_finetune
from src.agents.Agent_Med_NCA_Simple_finetuning import Agent_Med_NCA_finetuning
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.models.Model_M3DNCA_alive import M3DNCA_alive
from src.losses.LossFunctions import DiceFocalLoss_2
from src.utils.Experiment import Experiment
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
import time
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

config = [{
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
    'name': r'Med_NCA_Run123_Prostate_unsupervisedTest_pretrained', #12 or 13, 54 opt,
    'pretrained': r'Med_NCA_Run8_Prostate_unsupervisedTest_pretrained', 
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 3e-7,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 5,
    'evaluate_interval': 5,
    'n_epoch': 3000,
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [20, 10],
    'cell_fire_rate': 0.5,
    'batch_size': 6,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(64, 64), (256, 256)] ,
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': False,
    'rescale': True,
}
]

dataset = Dataset_NiiGz_3D(store=True, slice=2, augment="spike")
device = torch.device(config[0]['device'])
ca = MedNCA_finetune(channel_n=16, fire_rate=0.5, steps=64, device = "cuda:0", hidden_size=128, input_channels=1, output_channels=1, batch_duplication=1).to("cuda:0")
agent = Agent_Med_NCA_finetuning(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

print("--------------- TESTING HYP 99 ---------------")
hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, slice=2, size=(256, 256), imagePath=r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr", labelPath=r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr")
hyp99_test.exp = exp
agent.getAverageDiceScore(pseudo_ensemble=False, dataset=hyp99_test)

loss_function = DiceFocalLoss_2() 
agent.getAverageDiceScore(pseudo_ensemble=True)

# Get initial score
if exp.currentStep == 0:
    agent.getAverageDiceScore(pseudo_ensemble=False)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
agent.train(data_loader, loss_function)

start_time = time.perf_counter()
agent.getAverageDiceScore(pseudo_ensemble=False)
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"The function took {elapsed_time} seconds to execute.")

# %%

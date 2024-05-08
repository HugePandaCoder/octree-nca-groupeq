from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss

from src.models.Model_OctreeNCA import OctreeNCA
from src.utils.Study import Study

from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_UNet2D, EXP_M3DNCA, EXP_TransUNet, EXP_MEDNCA, EXP_OctreeNCA, EXP_BasicNCA

import matplotlib.pyplot as plt
import torch

###### Define basic configuration here ######

ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"

print(ProjectConfiguration.STUDY_PATH)

study_config = {
    'img_path': r"/local/scratch/jkalkhof/Data/MICCAI24/ChestX8_1k/images/",
    'label_path': r"/local/scratch/jkalkhof/Data/MICCAI24/ChestX8_1k/labels/",
    'name': r'Octree',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 10,
    'evaluate_interval': 5,
    'n_epoch': 10000,
    # Model
    'input_channels': 1,
    'output_channels': 1,
    # Data
    'input_size': [(256, 256)],
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': True,
    'rescale': False,

    ### TEMP
    
    'batch_size': 6,
}

study = Study(study_config)

from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
dataset = Nii_Gz_Dataset()

experiment = EXP_OctreeNCA().createExperiment(study_config, detail_config={'input_size':(256, 256)}, dataset=dataset)

assert isinstance(experiment.agent, MedNCAAgent)
assert isinstance(experiment.model, OctreeNCA)
with torch.no_grad():
    data = next(iter(experiment.data_loader))
    experiment.agent.prepare_data(data)
    inputs, targets = data['image'], data['label']
    gallery = experiment.model.create_inference_series(inputs)

    #convert to binary label
    gallery.append((gallery[-1] > 0.5).float())



plt.figure(figsize=(15, 5))
#plot all figures in gallery
for i, img in enumerate(gallery):
    plt.subplot(1, len(gallery)+1, i+1)
    plt.imshow(img[0, :, :, 1].cpu().numpy())
    plt.title(f"{img.shape[1]}x{img.shape[2]}", fontsize=8)
    plt.axis('off')


plt.subplot(1, len(gallery)+1, i+2)
plt.imshow(targets[0, 0, :, :].cpu().permute(1,0).numpy())
plt.title(f"ground truth", fontsize=8)
plt.axis('off')

plt.savefig("inference.png", bbox_inches='tight')
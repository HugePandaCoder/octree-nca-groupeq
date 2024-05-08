from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet2D, EXP_M3DNCA, EXP_TransUNet, EXP_MEDNCA, EXP_OctreeNCA, EXP_BasicNCA
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset

ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"

print(ProjectConfiguration.STUDY_PATH)


def setup_chest():
    study_config = {
        'img_path': r"/local/scratch/jkalkhof/Data/MICCAI24/ChestX8_1k/images/",
        'label_path': r"/local/scratch/jkalkhof/Data/MICCAI24/ChestX8_1k/labels/",
        'name': r'ChestX8_1k',
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
    dataset = Nii_Gz_Dataset()
    study.add_experiment(EXP_OctreeNCA().createExperiment(study_config, detail_config={'input_size':(256, 256)}, dataset=dataset))
    return study

def setup_prostate():
    study_config = {
        'img_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/imagesTr/",
        'label_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/labelsTr/",
        'name': r'Prostate',
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
        'input_size': [(256, 256, 16)],
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': True,
        'rescale': False,

        ### TEMP
        
        'batch_size': 6,
    }
    study = Study(study_config)
    dataset = Dataset_NiiGz_3D()
    study.add_experiment(EXP_OctreeNCA().createExperiment(study_config, detail_config={'input_size':(256, 256, 16)}, dataset=dataset))
    return study



def setup_hippocampus():
    study_config = {
        'img_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/imagesTr/",
        'label_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/labelsTr/",
        'name': r'Hippocampus',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 10,
        'evaluate_interval': 1,#5
        'n_epoch': 10000,
        # Model
        'input_channels': 1,
        'output_channels': 1,
        # Data
        'input_size': [(44, 60, 48)],#(44, 60, 48) -> (22, 30, 24) -> (11, 15, 12)
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': True,
        'rescale': False,

        ### TEMP
        
        'batch_size': 6,
    }
    study = Study(study_config)
    dataset = Dataset_NiiGz_3D()
    study.add_experiment(EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset))
    return study


if __name__ == "__main__":
    study = setup_hippocampus()
    study.run_experiments()
    study.eval_experiments()

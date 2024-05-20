from matplotlib import pyplot as plt
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet2D, EXP_M3DNCA, EXP_TransUNet, EXP_MEDNCA, EXP_OctreeNCA, EXP_BasicNCA
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch

ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"

print(ProjectConfiguration.STUDY_PATH)

PROSTATE_IMGS = r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/imagesTr/"
PROSTATE_LBLS = r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/labelsTr/"

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
        # Octree - specific
        #'octree_res_and_steps': [((256,256), 2), ((128,128), 2), ((64,64), 2), ((32,32), 2), ((16,16), 16)],
        #TODO implement this in the experiment and model!


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
        'input_size': [(320, 320, 24)], # (320, 320, 24) -> (160, 160, 12) -> (80, 80, 12) -> (40, 40, 12) -> (20, 20, 12)
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': True,
        'rescale': False,
        # Octree - specific
        'octree_res_and_steps': [((320,320,24), 2), ((160,160,12), 2), ((80,80,12), 2), ((40,40,12), 2), ((20,20,12), 3)],

        ### TEMP
        
        'batch_size': 1,
    }
    study = Study(study_config)
    dataset = Dataset_NiiGz_3D()
    study.add_experiment(EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset))
    return study

def setup_prostate2():
    study_config = {
        'img_path': r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/imagesTr/",
        'label_path': r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/labelsTr/",
        'name': r'Prostate26',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 10,
        'evaluate_interval': 10,
        'n_epoch': 1500,
        # Model
        'input_channels': 1,
        'output_channels': 1,
        'hidden_size': 64,
        'train_model':1,
        'channel_n': 16,
        'kernel_size': 7,
        # Data
        'input_size': [(320, 320, 24)], # (320, 320, 24) -> (160, 160, 12) -> (80, 80, 12) -> (40, 40, 12) -> (20, 20, 12)
        
        'data_split': [1.0, 0, 0.0], 
        'keep_original_scale': False,
        'rescale': True,
        # Octree - specific
        'octree_res_and_steps': [((320,320,24), 40), ((160,160,12), 0), ((80,80,6), 20), ((40,40,6), 0), ((20,20,6), 0)],
        'separate_models': True,
        # (160, 160, 12) <- (160, 160, 12) <- (80, 80, 12) <- (40, 40, 12) <- (20, 20, 12)
        'patch_sizes':[((80, 80, 6)), None, None, None, None],
        #'patch_sizes': [None] *5,
        ### TEMP
        

        'compile': False,
        'batch_size': 4,
        'batch_duplication': 2,
    }
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.autograd.set_detect_anomaly(True)
    study = Study(study_config)
    dataset = Dataset_NiiGz_3D()
    exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset)
    study.add_experiment(exp)
    
    hyp99_test = Dataset_NiiGz_3D_customPath(resize=True, size=(320, 320, 24), imagePath=r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/imagesTs", labelPath=r"/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/labelsTs")
    hyp99_test.exp = exp
    study.my_custom_evaluation_set = hyp99_test
    def evaluate():
        print("RUNNING CUSTOM EVALUATION")

        exp.agent.getAverageDiceScore(pseudo_ensemble=True, dataset=study.my_custom_evaluation_set)
    
    study.eval_experiments = evaluate
    return study


def setup_hippocampus():
    study_config = {
        'img_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/imagesTr/",
        'label_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/labelsTr/",
        'name': r'Hippocampus4',
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
        'input_size': [(44, 60, 48)],#(44, 60, 48) -> (22, 30, 24) -> (11, 15, 12)
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': True,
        'rescale': False,
        # Octree - specific
        'octree_res_and_steps': [((44,60,48), 2), ((22,30,24), 2), ((11,15,12), 16)],
        'separate_models': True,

        ### TEMP
        
        'batch_size': 6,
    }
    study = Study(study_config)
    dataset = Dataset_NiiGz_3D()
    study.add_experiment(EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset))
    return study


def setup_hippocampus2():
    study_config = {
        'img_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/imagesTr/",
        'label_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/labelsTr/",
        'name': r'Hippocampus2',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr': 16e-4,
        'lr_gamma': 0.9999,
        'betas': (0.5, 0.5),
        # Training
        'save_interval': 10,
        'evaluate_interval': 5,
        'n_epoch': 10000,
        # Model
        'input_channels': 1,
        'output_channels': 1,
        # Data
        'input_size': [(44, 60, 48)],#(44, 60, 48) -> (22, 30, 24) -> (11, 15, 12)
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': True,
        'rescale': False,
        # Octree - specific
        'octree_res_and_steps': [((44,60,48), 2), ((22,30,24), 2), ((11,15,12), 16)],


        ### TEMP
        
        'batch_size': 6, #probably make this a lot bigger
    }
    study = Study(study_config)
    dataset = Dataset_NiiGz_3D()
    study.add_experiment(EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset))
    return study


def train_hippocampus_baseline():
    import torch
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
    from src.models.Model_BasicNCA3D import BasicNCA3D
    from src.losses.LossFunctions import DiceFocalLoss
    from src.utils.Experiment import Experiment
    from src.agents.Agent_M3D_NCA import Agent_M3D_NCA
    config = [{
        'img_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/imagesTr/",
        'label_path': r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task04_Hippocampus/labelsTr/",
        'name': r'M3D_NCA_Run8',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr': 16e-4,
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 10,
        'evaluate_interval': 10,
        'n_epoch': 3000,
        'batch_duplication': 1,
        # Model
        'channel_n': 16,        # Number of CA state channels
        'inference_steps': [10, 10],
        'cell_fire_rate': 0.5,
        'batch_size': 4,
        'input_channels': 1,
        'output_channels': 1,
        'hidden_size': 64,
        'train_model':1,
        # Data
        'input_size': [(16, 16, 13),(64, 64, 52)], # 
        'scale_factor': 4,
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': True,
        'rescale': True,
    }]

    dataset = Dataset_NiiGz_3D()
    device = torch.device(config[0]['device'])
    ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels']).to(device)
    ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
    ca = [ca1, ca2]
    agent = Agent_M3D_NCA(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    loss_function = DiceFocalLoss()  
    agent.train(data_loader, loss_function)
    agent.getAverageDiceScore()


def train_prostate_baseline():
    import torch
    from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
    from src.models.Model_BasicNCA3D import BasicNCA3D
    from src.losses.LossFunctions import DiceFocalLoss
    from src.utils.Experiment import Experiment
    from src.agents.Agent_M3D_NCA import Agent_M3D_NCA
    config = [{
        'img_path': PROSTATE_IMGS,
        'label_path': PROSTATE_LBLS,
        'model_path': r'Models/M3D_NCA_Run1',
        'name': 'prostate_baseline',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr': 16e-4,
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 10,
        'evaluate_interval': 10,
        'n_epoch': 3000,
        'batch_duplication': 1,
        # Model
        'channel_n': 16,        # Number of CA state channels
        'inference_steps': [20, 40],
        'cell_fire_rate': 0.5,
        'batch_size': 4,
        'input_channels': 1,
        'output_channels': 1,
        'hidden_size': 64,
        'train_model':1,
        # Data
        'input_size': [(80, 80, 6),(320, 320, 24)], # 
        'scale_factor': 4,
        'data_split': [0.7, 0, 0.3], 
        'keep_original_scale': False,
        'rescale': True,
    }
    ]
    dataset = Dataset_NiiGz_3D()
    device = torch.device(config[0]['device'])
    ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels']).to(device)
    ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
    ca = [ca1, ca2]
    agent = Agent_M3D_NCA(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    loss_function = DiceFocalLoss()
    agent.train(data_loader, loss_function)


if __name__ == "__main__":
    study = setup_prostate2()
    #study = setup_hippocampus()
    #figure = octree_vis.visualize(study.experiments[0], study.my_custom_evaluation_set)
    #plt.savefig("inference_test.png", bbox_inches='tight')
    study.run_experiments()
    study.eval_experiments()


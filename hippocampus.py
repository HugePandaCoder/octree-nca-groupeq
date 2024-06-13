from matplotlib import pyplot as plt
from src.datasets.BatchgeneratorsDataLoader import get_batchgenerators_dataloader_dataset
from src.datasets.BatchgeneratorsDatasetWrapperDataset import get_batchgenerators_dataset
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.DataAugmentations import get_augmentation_dataset
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet2D, EXP_M3DNCA, EXP_TransUNet, EXP_MEDNCA, EXP_OctreeNCA, EXP_BasicNCA
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed

HIPPOCAMPUS_SPLIT_FILE = r"/local/scratch/clmn1/octree_study/Experiments/hippocampus2_OctreeNCA3D/data_split.dt"
HIPPOCAMPUS_SPLIT = pkl.load(open(HIPPOCAMPUS_SPLIT_FILE, "rb"))

def setup_hippocampus():
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    study_config = {
        'img_path': r"/local/scratch/clmn1/data/Task97_DecathHip/custom_imagesTr",
        'label_path': r"/local/scratch/clmn1/data/Task97_DecathHip/labelsTr_/",
        'name': r'hippocampus2',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr_gamma': 0.9999**8,
        'lr': 0.0016,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 50,
        'evaluate_interval': 200,
        'n_epoch': 2000,
        # Model
        'input_channels': 1,
        'output_channels': 1,
        'hidden_size': 64,
        'train_model':1,
        'channel_n': 16,
        'kernel_size': [3, 3, 3],
        # Data
        'input_size': [(44, 60, 48)], #(44, 60, 48) -> (22, 30, 24) -> (11, 15, 12)
        
        'data_split': [0.7, 0, 0.3],
        'keep_original_scale': True,
        'rescale': True,
        # Octree - specific
        'octree_res_and_steps': [((44, 60, 48), 5), ((22, 30, 24), 5), ((11, 15, 12), 20)],
        'separate_models': True,
        # (160, 160, 12) <- (160, 160, 12) <- (80, 80, 12) <- (40, 40, 12) <- (20, 20, 12)
        'patch_sizes':[None, None, None, None, None],
        #'patch_sizes': [None] *5,
        ### TEMP
        'gradient_accumulation': False,
        'train_quality_control': False, #or "NQM" or "MSE"

        'compile': False,
        'data_parallel': False,
        'batch_size': 3,
        'batch_duplication': 1,
        'num_workers': 8,
        'update_lr_per_epoch': True, # is false by default
         # TODO batch duplication per level could be helpful as the levels with a patchsize are much more stochastic than others.
         # Alternativly, train for more epochs and slower weight decay or iterate through all epochs (deterministically, no random sampling of patches)
        'also_eval_on_train': True,
        'num_steps_per_epoch': None, #default is None
        'train_data_augmentations': True,
        'track_gradient_norm': True,
        'batchgenerators': True, 
        'loss_weighted_patching': True,# default false, train on the patch that has the highest loss in the previous epoch
        # TODO 'lambda_dice_loss'
        # TODO maybe diffulty weighted sampling
        # TODO more data augmentations
        # TODO different weight initializations
        # TODO maybe mask CE loss on correctly segmented areas
        # TODO try adam params (0.5, 0.5)
        'difficulty_weighted_sampling': False, #default is False. Difficulty is evaluated at every 'evaluate_interval' epoch. Also, 'also_eval_on_train' _must_ be True

        'optimizer': "Adam",# default is "Adam"
        'sgd_momentum': 0.99,
        'sgd_nesterov': True,

        'scheduler': "exponential",#default is exponential
        'polynomial_scheduler_power': 1.8,

        'find_best_model_on': 'train', # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
        'always_eval_in_last_epochs': 300, #default is None
    }
    if study_config['difficulty_weighted_sampling']:
        assert study_config['batchgenerators']
        assert study_config['also_eval_on_train']
        assert study_config['num_steps_per_epoch'] is not None
    #assert (study_config['num_steps_per_epoch'] is not None) == study_config['batchgenerators']
    study = Study(study_config)

    if study_config['batchgenerators']:
        #dataset = get_batchgenerators_dataloader_dataset(Dataset_NiiGz_3D, study_config['train_data_augmentations'], 
        #                                                 study_config['num_steps_per_epoch'], study_config['batch_size'],
        #                                                 study_config['num_workers'])()
        dataset = get_batchgenerators_dataset(Dataset_NiiGz_3D, study_config['num_workers'], 
                                              study_config['num_steps_per_epoch'], study_config['batch_size'],
                                              study_config['difficulty_weighted_sampling'])()
    else:
        if study_config['train_data_augmentations']:
            dataset = get_augmentation_dataset(Dataset_NiiGz_3D)()
        else:
            dataset = Dataset_NiiGz_3D()
    exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset)
    study.add_experiment(exp)


    for split in "train", "val", "test":
        if not exp.data_split.get_images(split) == HIPPOCAMPUS_SPLIT.get_images(split):
            print("SPLIT MISMATCH")
            os.remove(f"/local/scratch/clmn1/octree_study/Experiments/{study_config['name']}_OctreeNCA3D/data_split.dt")
            os.symlink(HIPPOCAMPUS_SPLIT_FILE, f"/local/scratch/clmn1/octree_study/Experiments/{study_config['name']}_OctreeNCA3D/data_split.dt")
            print("restart experiment!")
            exit()
            
    for split in "train", "val", "test":
        assert exp.data_split.get_images(split) == HIPPOCAMPUS_SPLIT.get_images(split)

    return study

def setup_baseline():
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    study_config = {
        'img_path': r"/local/scratch/clmn1/data/Task97_DecathHip/custom_imagesTr",
        'label_path': r"/local/scratch/clmn1/data/Task97_DecathHip/labelsTr_/",
        'name': r'hippocampus_baseline1',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr_gamma': 0.9999**8,
        'lr': 0.0016,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 50,
        'evaluate_interval': 200,
        'n_epoch': 2000,
        # Model
        'input_channels': 1,
        'output_channels': 1,
        'hidden_size': 64,
        'train_model':1,
        'channel_n': 16,
        'kernel_size': [7],
        # Data
        'input_size': [(44, 60, 48)], #(44, 60, 48) -> (22, 30, 24) -> (11, 15, 12)
        
        'data_split': [0.7, 0, 0.3],
        'keep_original_scale': True,
        'rescale': True,
        # Octree - specific
        'octree_res_and_steps': [((44, 60, 48), 30)],
        'separate_models': True,
        # (160, 160, 12) <- (160, 160, 12) <- (80, 80, 12) <- (40, 40, 12) <- (20, 20, 12)
        'patch_sizes':[None, None, None, None, None],
        #'patch_sizes': [None] *5,
        ### TEMP
        'gradient_accumulation': False,
        'train_quality_control': False, #or "NQM" or "MSE"

        'compile': True,
        'data_parallel': False,
        'batch_size': 3,
        'batch_duplication': 1,
        'num_workers': 8,
        'update_lr_per_epoch': True, # is false by default
         # TODO batch duplication per level could be helpful as the levels with a patchsize are much more stochastic than others.
         # Alternativly, train for more epochs and slower weight decay or iterate through all epochs (deterministically, no random sampling of patches)
        'also_eval_on_train': True,
        'num_steps_per_epoch': None, #default is None
        'train_data_augmentations': True,
        'track_gradient_norm': True,
        'batchgenerators': True, 
        'loss_weighted_patching': False,# default false, train on the patch that has the highest loss in the previous epoch
        # TODO 'lambda_dice_loss'
        # TODO maybe diffulty weighted sampling
        # TODO more data augmentations
        # TODO different weight initializations
        # TODO maybe mask CE loss on correctly segmented areas
        # TODO try adam params (0.5, 0.5)
        'difficulty_weighted_sampling': False, #default is False. Difficulty is evaluated at every 'evaluate_interval' epoch. Also, 'also_eval_on_train' _must_ be True

        'optimizer': "Adam",# default is "Adam"
        'sgd_momentum': 0.99,
        'sgd_nesterov': True,

        'scheduler': "exponential",#default is exponential
        'polynomial_scheduler_power': 1.8,

        'find_best_model_on': 'train', # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
        'always_eval_in_last_epochs': 300, #default is None
    }
    if study_config['difficulty_weighted_sampling']:
        assert study_config['batchgenerators']
        assert study_config['also_eval_on_train']
        assert study_config['num_steps_per_epoch'] is not None
    #assert (study_config['num_steps_per_epoch'] is not None) == study_config['batchgenerators']
    study = Study(study_config)

    if study_config['batchgenerators']:
        #dataset = get_batchgenerators_dataloader_dataset(Dataset_NiiGz_3D, study_config['train_data_augmentations'], 
        #                                                 study_config['num_steps_per_epoch'], study_config['batch_size'],
        #                                                 study_config['num_workers'])()
        dataset = get_batchgenerators_dataset(Dataset_NiiGz_3D, study_config['num_workers'], 
                                              study_config['num_steps_per_epoch'], study_config['batch_size'],
                                              study_config['difficulty_weighted_sampling'])()
    else:
        if study_config['train_data_augmentations']:
            dataset = get_augmentation_dataset(Dataset_NiiGz_3D)()
        else:
            dataset = Dataset_NiiGz_3D()
    exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset=dataset)
    study.add_experiment(exp)


    for split in "train", "val", "test":
        if not exp.data_split.get_images(split) == HIPPOCAMPUS_SPLIT.get_images(split):
            print("SPLIT MISMATCH")
            os.remove(f"/local/scratch/clmn1/octree_study/Experiments/{study_config['name']}_OctreeNCA3D/data_split.dt")
            os.symlink(HIPPOCAMPUS_SPLIT_FILE, f"/local/scratch/clmn1/octree_study/Experiments/{study_config['name']}_OctreeNCA3D/data_split.dt")
            print("restart experiment!")
            exit()
            
    for split in "train", "val", "test":
        assert exp.data_split.get_images(split) == HIPPOCAMPUS_SPLIT.get_images(split)

    return study


if __name__ == "__main__":
    study = setup_baseline()
    #figure = octree_vis.visualize(study.experiments[0], sample_id="prostate_13.nii.gz")
    #plt.savefig("inference_test.png", bbox_inches='tight')
    study.run_experiments()
    study.eval_experiments()
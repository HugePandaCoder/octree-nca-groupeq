from matplotlib import pyplot as plt
from src.datasets.BatchgeneratorsDataLoader import get_batchgenerators_dataloader_dataset
from src.datasets.BatchgeneratorsDatasetWrapperDataset import get_batchgenerators_dataset
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset_3D_customPath import Dataset_NiiGz_3D_customPath
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss
from src.utils.BaselineConfigs import EXP_OctreeNCA3D_superres
from src.utils.DataAugmentations import get_augmentation_dataset
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"

print(ProjectConfiguration.STUDY_PATH)

PROSTATE_IMGS = r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/imagesTr/"
PROSTATE_LBLS = r"/local/scratch/clmn1/cardiacProstate/nnUnet_raw_data_base/Task05_Prostate/labelsTr/"

PROSTATE_49_SPLIT_FILE = r"/local/scratch/clmn1/octree_study/Experiments/Prostate49_OctreeNCA3D/data_split.dt"
PROSTATE_49_SPLIT = pkl.load(open(PROSTATE_49_SPLIT_FILE, "rb"))


#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
study_config = {
    'img_path': r"/local/scratch/jkalkhof/Data/Prostate_MEDSeg/imagesTr/",
    'label_path': r"/local/scratch/jkalkhof/Data/Prostate_MEDSeg/labelsTr/",
    'name': r'Prostate49_superres_2_residual',
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
    'kernel_size': [3, 3, 3, 3, 3],
    # Data
    'input_size': [(320, 320, 24)], # (320, 320, 24) -> (160, 160, 12) -> (80, 80, 12) -> (40, 40, 12) -> (20, 20, 12)
    
    'data_split': [0.7, 0, 0.3],
    'keep_original_scale': True,
    'rescale': True,
    # Octree - specific
    'octree_res_and_steps': [((320,320,24), 5), ((160,160,12), 5), ((80,80,6), 5), ((40,40,6), 5), ((20,20,6), 20)],
    'separate_models': True,
    # (160, 160, 12) <- (160, 160, 12) <- (80, 80, 12) <- (40, 40, 12) <- (20, 20, 12)
    'patch_sizes':[(80, 80, 6), (80, 80, 6), None, None, None],
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

    'find_best_model_on': None, # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
    'always_eval_in_last_epochs': None, #default is None

    
    'apply_ema': True,
    'ema_decay': 0.99,
    'ema_update_per': "epoch",

    'superres_factor': 4,



}
study = Study(study_config)


exp = EXP_OctreeNCA3D_superres().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)


for split in "train", "val", "test":
    if not exp.data_split.get_images(split) == PROSTATE_49_SPLIT.get_images(split):
        print("SPLIT MISMATCH")
        os.remove(f"/local/scratch/clmn1/octree_study/Experiments/{study_config['name']}_OctreeNCA3D/data_split.dt")
        os.symlink(PROSTATE_49_SPLIT_FILE, f"/local/scratch/clmn1/octree_study/Experiments/{study_config['name']}_OctreeNCA3D/data_split.dt")
        print("restart experiment!")
        exit()
        
for split in "train", "val", "test":
    assert exp.data_split.get_images(split) == PROSTATE_49_SPLIT.get_images(split)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])



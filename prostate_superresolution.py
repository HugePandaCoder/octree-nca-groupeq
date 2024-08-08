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


study_config = {
    'experiment.name': r'prostate_superres_2_test',
    'experiment.description': "OctreeNCASuperresolution",
    'experiment.data_split': [0.7, 0, 0.3],
    'experiment.save_interval': 50,
    'experiment.device': "cuda:0",

    'experiment.logging.also_eval_on_train': True,
    'experiment.logging.track_gradient_norm': True,
    'experiment.logging.evaluate_interval': 1,

    'experiment.dataset.img_path': r"/local/scratch/jkalkhof/Data/Prostate_MEDSeg/imagesTr/",
    'experiment.dataset.label_path': r"/local/scratch/jkalkhof/Data/Prostate_MEDSeg/labelsTr/",
    'experiment.dataset.keep_original_scale': True,
    'experiment.dataset.rescale': True,
    'experiment.dataset.input_size': [320, 320, 24],
    'experiment.dataset.patchify': False,


    'experiment.task': "super_resolution",
    'experiment.task.factor': 4,
    'experiment.task.train_on_residual': True,


    'performance.compile': False,
    'performance.data_parallel': False,
    'performance.num_workers': 8,
    'performance.unlock_CPU': True,
    'performance.inplace_operations': True,


    'trainer.optimizer': "torch.optim.AdamW",
    'trainer.optimizer.lr': 1e-3,
    'trainer.lr_scheduler': "torch.optim.lr_scheduler.CosineAnnealingLR",
    'trainer.lr_scheduler.T_max': 2000,
    'trainer.update_lr_per_epoch': True,
    'trainer.losses': ["torch.nn.L1Loss"],
    'trainer.loss_weights': [1e2],
    'trainer.normalize_gradients': "layerwise", # all, layerwise, none

    'trainer.n_epochs': 2000,
    'trainer.num_steps_per_epoch': None,
    'trainer.batch_size': 3,
    'trainer.batch_duplication': 1,

    'trainer.find_best_model_on': None,
    'trainer.always_eval_in_last_epochs': None,

    'trainer.ema': True,
    'trainer.ema.decay': 0.99,
    'trainer.ema.update_per': "epoch",

    'trainer.datagen.batchgenerators': True,
    'trainer.datagen.augmentations': True,
    'trainer.datagen.difficulty_weighted_sampling': False,

    'trainer.gradient_accumulation': False,
    'trainer.train_quality_control': False, #or "NQM" or "MSE"



    'model.channel_n': 16,
    'model.fire_rate': 0.5,
    'model.input_channels': 1,
    'model.output_channels': 1,
    'model.kernel_size': [3, 3, 3, 3, 3],
    'model.hidden_size': 64,
    'model.batchnorm_track_running_stats': False,

    'model.train.patch_sizes': [[80, 80, 6], [80, 80, 6], None, None, None],
    'model.train.loss_weighted_patching': False,

    'model.octree.res_and_steps': [[[320,320,24], 5], [[160,160,12], 5], [[80,80,6], 5], [[40,40,6], 5], [[20,20,6], 20]],
    'model.octree.separate_models': True,

    'model.vitca': False, # TODO 3D ViTCA is not implemented yet!
    'model.vitca.depth': 1,
    'model.vitca.heads': 4,
    'model.vitca.mlp_dim': 64,
    'model.vitca.dropout': 0.0,
    'model.vitca.positional_embedding': 'vit_handcrafted', #'vit_handcrafted', 'nerf_handcrafted', 'learned', or None for no positional encoding
    'model.vitca.embed_cells': True,
    'model.vitca.embed_dim': 128,
    'model.vitca.embed_dropout': 0.0,
}
study = Study(study_config)


exp = EXP_OctreeNCA3D_superres().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)


for split in "train", "val", "test":
    if not exp.data_split.get_images(split) == PROSTATE_49_SPLIT.get_images(split):
        print("SPLIT MISMATCH")
        os.remove(f"/local/scratch/clmn1/octree_study/Experiments/{study_config['experiment.name']}_{study_config['experiment.description']}/data_split.pkl")
        os.symlink(PROSTATE_49_SPLIT_FILE, f"/local/scratch/clmn1/octree_study/Experiments/{study_config['experiment.name']}_{study_config['experiment.description']}/data_split.pkl")
        print("restart experiment!")
        exit()
        
for split in "train", "val", "test":
    assert exp.data_split.get_images(split) == PROSTATE_49_SPLIT.get_images(split)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])



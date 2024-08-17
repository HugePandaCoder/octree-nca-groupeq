from matplotlib import pyplot as plt
import configs.default
import configs.models
import configs.tasks
import configs.tasks.segmentation
import configs.trainers
import configs.trainers.vitca
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_OctreeNCA3D_superres
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed

import torchio as tio

import configs
from src.utils.convert_to_cluster import convert_paths_to_cluster_paths, maybe_convert_paths_to_cluster_paths

print(ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'new_prostate_overkill5',
    'experiment.description': "OctreeNCASegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.prostate.prostate_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.prostate.prostate_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['trainer.ema'] = False
study_config['performance.compile'] = True
study_config['model.train.loss_weighted_patching'] = True

study_config['trainer.find_best_model_on'] = "train"
study_config['trainer.always_eval_in_last_epochs'] = 300

study_config['model.channel_n'] = 24
study_config['model.hidden_size'] = 100
study_config['model.kernel_size'] = [3, 3, 3, 3, 7]
study_config['model.octree.res_and_steps'] = [[[320,320,24], 20], [[160,160,12], 20], [[80,80,6], 20], [[40,40,6], 20], [[20,20,6], 40]]

study_config = maybe_convert_paths_to_cluster_paths(study_config)

study = Study(study_config)

ood_augmentation = None
severity = 6
#ood_augmentation = tio.RandomGhosting(num_ghosts=severity, intensity=0.25 * severity)

exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments(ood_augmentation=ood_augmentation)
#figure = octree_vis.visualize(study.experiments[0])



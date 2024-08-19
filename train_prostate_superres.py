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

print(ProjectConfiguration.STUDY_PATH)
import socket
print(socket.gethostname(), torch.cuda.get_device_name())


study_config = {
    'experiment.name': r'new_prostate_superres_overkill_2_dev',
    'experiment.description': "OctreeNCASuperres",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.prostate.prostate_model_config
study_config = study_config | configs.trainers.vitca.vitca_trainer_config
study_config = study_config | configs.datasets.prostate.prostate_dataset_config
study_config = study_config | configs.tasks.superres.superres_task_config
study_config = study_config | configs.default.default_config

study_config['trainer.ema'] = False
study_config['performance.compile'] = False
study_config['model.train.loss_weighted_patching'] = False

study_config['trainer.find_best_model_on'] = False
study_config['trainer.always_eval_in_last_epochs'] = None

study_config['model.channel_n'] = 24
study_config['model.hidden_size'] = 100
study_config['model.kernel_size'] = [3, 3, 3, 3, 7]
study_config['model.octree.res_and_steps'] = [[[320,320,24], 20], [[160,160,12], 20], [[80,80,6], 20], [[40,40,6], 20], [[20,20,6], 40]]

study_config['experiment.logging.evaluate_interval'] = 1
#study_config['model.train.patch_sizes'] = [[40, 40, 3], [40, 40, 3], None, None, None]
#study_config['model.channel_n'] = 12

study = Study(study_config)

ood_augmentation = None
output_name = None
severity = 6
#ood_augmentation = tio.RandomGhosting(num_ghosts=severity, intensity=0.5 * severity)
if ood_augmentation != None:
    output_name = f"{ood_augmentation.__class__.__name__}_{severity}.csv"

exp = EXP_OctreeNCA3D_superres().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments(ood_augmentation=ood_augmentation, output_name=output_name)
#figure = octree_vis.visualize(study.experiments[0])


#study.eval_experiments_ood()

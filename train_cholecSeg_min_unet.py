from matplotlib import pyplot as plt
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_min_UNet
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
import octree_vis, os, torch, shutil
import pickle as pkl

import torchio as tio

import configs

#ProjectConfiguration.STUDY_PATH = r"clmn1/octree_study_dev/"

print(ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'cholec',
    'experiment.description': "MinUNetSegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.min_unet.min_unet_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config


study_config['model.encoder_name'] = "mobilenet_v2"

study_config['experiment.name'] += f"_{study_config['model.arch']}_{study_config['model.encoder_name']}_0"


study_config['trainer.ema'] = False
study_config['performance.compile'] = False
study_config['experiment.logging.also_eval_on_train'] = False

study_config['trainer.n_epochs'] = 400
study_config['trainer.batch_size'] = 10
study_config['experiment.dataset.input_size'] = [256, 448, 80]
study_config['trainer.num_steps_per_epoch'] = 100
study_config['experiment.logging.evaluate_interval'] = 2001#study_config['trainer.n_epochs'] + 1


#study_config['experiment.save_interval'] = 1
#study_config['experiment.logging.evaluate_interval'] = 1
#study_config['trainer.n_epochs'] = 5 
#study_config['trainer.num_steps_per_epoch'] = 5


study = Study(study_config)

ood_augmentation = None
output_name = None
severity = 6
#ood_augmentation = tio.RandomGhosting(num_ghosts=severity, intensity=0.5 * severity)
if ood_augmentation != None:
    output_name = f"{ood_augmentation.__class__.__name__}_{severity}.csv"


exp = EXP_min_UNet().createExperiment(study_config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                                'slice_wise': True
                                            })
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments(ood_augmentation=ood_augmentation, output_name=output_name)
#figure = octree_vis.visualize(study.experiments[0])

#study.eval_experiments_ood()


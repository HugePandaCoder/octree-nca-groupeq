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


import configs.datasets.cholec, configs.datasets.prostate, configs.datasets.peso
import configs.models.prostate, configs.models.prostate_vitca
import configs.trainers.vitca, configs.trainers.nca
import configs.tasks.segmentation, configs.tasks.extrapolation, configs.tasks.superres
import configs.default

ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"

print(ProjectConfiguration.STUDY_PATH)

PROSTATE_49_SPLIT_FILE = r"/local/scratch/clmn1/octree_study/Experiments/Prostate49_OctreeNCA3D/data_split.dt"
PROSTATE_49_SPLIT = pkl.load(open(PROSTATE_49_SPLIT_FILE, "rb"))

study_config = {
    'experiment.name': r'new_prostate_4',
    'experiment.description': "OctreeNCASegmentation",
}
study_config = study_config | configs.models.prostate.prostate_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.prostate.prostate_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

#study_config['trainer.ema'] = False
study_config['performance.compile'] = True
#study_config['trainer.loss_weights'] = [1e2]
study_config['model.train.loss_weighted_patching'] = True

study = Study(study_config)


exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})
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



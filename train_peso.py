from matplotlib import pyplot as plt
import configs
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.datasets.Dataset_DAVIS import Dataset_DAVIS
from src.datasets.Dataset_PESO import Dataset_PESO
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.BaselineConfigs import EXP_OctreeNCA, EXP_OctreeNCA3D, EXP_OctreeNCA3D_superres
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.datasets.png_seg_Dataset import png_seg_Dataset
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import octree_vis, os, torch, shutil
import pickle as pkl
from src.datasets.Dataset_CholecSeg import Dataset_CholecSeg
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed

import torchio as tio
pc.STUDY_PATH = r"clmn1/octree_study_dev/"

print(pc.STUDY_PATH)

study_config = {
    'experiment.name': r'pesoLargeGroupNormRetrain2',
    'experiment.description': "OctreeNCA2DSegmentation",

    'model.output_channels': 1,
}
study_config = study_config | configs.models.peso.peso_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.peso.peso_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['experiment.logging.also_eval_on_train'] = False
study_config['experiment.logging.evaluate_interval'] = study_config['trainer.n_epochs']+1
study_config['experiment.task.score'] = ["src.scores.PatchwiseDiceScore.PatchwiseDiceScore",
                                         "src.scores.PatchwiseIoUScore.PatchwiseIoUScore"]

study_config['model.octree.res_and_steps'] = [[[320,320], 20], [[160,160], 20], [[80,80], 20], [[40,40], 20], [[20,20], 40]]


study_config['trainer.losses'] = ["src.losses.DiceLoss.DiceLoss", "src.losses.BCELoss.BCELoss"]
study_config['trainer.losses.parameters'] = [{}, {}]
study_config['trainer.loss_weights'] = [1.0, 1.0]

study_config['model.normalization'] = "group"



#study_config['experiment.logging.evaluate_interval'] = 1
#study_config['trainer.num_steps_per_epoch'] = 2



#study_config['trainer.datagen.difficulty_weighted_sampling'] = True
#study_config['trainer.datagen.difficulty_dict'] = "/local/scratch/clmn1/octree_study_new/Experiments/pesoLarge_loss2_contd_OctreeNCA2DSegmentation/computed_disagreement.pkl"

study_config["experiment.dataset.return_background_class"] = True
study_config["model.output_channels"] += 1
study_config['trainer.losses'] = ["src.losses.DiceLoss.nnUNetSoftDiceLoss", "torch.nn.CrossEntropyLoss"]
study_config['trainer.losses.parameters'] = [{"apply_nonlin": None, "batch_dice": True, "do_bg": False, "smooth": 1e-05}, {}]
study_config['model.apply_nonlin'] = "torch.nn.Softmax(dim=1)"

study = Study(study_config)

exp = EXP_OctreeNCA().createExperiment(study_config, detail_config={}, dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.patches_path']),
                                                            'patch_size': study_config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.img_path']),
                                                            'img_level': study_config['experiment.dataset.img_level'],
                                                            'return_background_class': study_config.get('experiment.dataset.return_background_class', False),
                                                      })
study.add_experiment(exp)

study.run_experiments()


study.eval_experiments(export_prediction=True)

#figure = octree_vis.visualize(study.experiments[0], sample_id=("pds_14_HE", 14400, 24320), per_step=True)
#figure.savefig("visualize_neg.pdf", bbox_inches='tight')
#figure = octree_vis.visualize(study.experiments[0], sample_id=("pds_28_HE", 12160, 38400), per_step=False)
#figure.savefig("visualize_neg2.png", bbox_inches='tight')
#figure = octree_vis.visualize(study.experiments[0], sample_id=("pds_10_HE", 11840, 38400), per_step=True)
#figure.savefig("visualize_pos.pdf", bbox_inches='tight')
#figure = octree_vis.visualize(study.experiments[0], sample_id=("pds_10_HE", 11840+320, 38400+320), per_step=True)
#figure.savefig("visualize_pos2.pdf", bbox_inches='tight')

assert(isinstance(exp.agent, MedNCAAgent))
#exp.agent.getAverageDiceScore(pseudo_ensemble=True, ood_augmentation=None, output_name=None, export_prediction=True)

#save figure to file
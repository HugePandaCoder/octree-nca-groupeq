
import os

from matplotlib import pyplot as plt
from src.datasets.Dataset_PESO import Dataset_PESO
from src.utils.Study import Study
from src.utils.BaselineConfigs import EXP_OctreeNCA, EXP_OctreeNCA2D_extrapolation
import octree_vis
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
import configs

print("Study Path:", pc.STUDY_PATH)

study_config = {
        'experiment.name': r'peso_extrapolation_vitca_top_only_masked_2',
        'experiment.description': "OctreeNCAExtrapolation",
}

study_config = study_config | configs.models.peso_vitca.peso_vitca_model_config
study_config = study_config | configs.trainers.vitca.vitca_trainer_config
study_config = study_config | configs.datasets.peso.peso_dataset_config
study_config = study_config | configs.tasks.extrapolation.extrapolation_task_config
study_config = study_config | configs.default.default_config

study_config['model.output_channels'] = 3
study_config['experiment.logging.also_eval_on_train'] = False
study_config['trainer.losses'] = ["src.losses.MaskedL1Loss.MaskedL1Loss", "src.losses.OverflowLoss.MaskedOverflowLoss"]
study_config['trainer.loss_weights'] = [1e2, 1e2]
study_config['experiment.logging.evaluate_interval']= 2001


study_config['model.octree.res_and_steps'] = [[[160, 160], 5], [[80, 80], 10], [[40, 40], 10], [[20,20], 10], [[10, 10], 20]]
study_config['experiment.dataset.input_size'] = [160, 160]

study_config['performance.compile'] = False

study_config['experiment.task.margin'] = 30
study_config['experiment.task.direction'] = "top"


study = Study(study_config)

###### Define specific model setups here and save them in list ######

study.add_experiment(EXP_OctreeNCA2D_extrapolation().createExperiment(study_config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.patches_path']),
                                                            'patch_size': study_config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, study_config['experiment.dataset.img_path']),
                                                            'img_level': study_config['experiment.dataset.img_level']
                                                      }))

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])
plt.show()
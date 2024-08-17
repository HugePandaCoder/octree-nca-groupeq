from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D
import octree_vis
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed


import configs
from src.utils.convert_to_cluster import maybe_convert_paths_to_cluster_paths

print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'new_cholec_seg_2_cluster',
    'experiment.description': "OctreeNCASegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.cholec.cholec_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['performance.compile'] = True
study_config['also_eval_on_train'] = False
study_config['model.channel_n'] = 20

study_config = maybe_convert_paths_to_cluster_paths(study_config)
study = Study(study_config)

exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                                'use_max_sequence_length_in_eval': False
                                            })
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])

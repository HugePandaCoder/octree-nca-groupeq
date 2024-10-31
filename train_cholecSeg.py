from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA3D
import octree_vis
from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
import configs

#ProjectConfiguration.STUDY_PATH = r"clmn1/octree_study_dev/"
print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
    'experiment.name': r'cholecGN',
    'experiment.description': "OctreeNCASegmentation",

    'model.output_channels': 5,
}
study_config = study_config | configs.models.cholec.cholec_model_config
study_config = study_config | configs.trainers.nca.nca_trainer_config
study_config = study_config | configs.datasets.cholec.cholec_dataset_config
study_config = study_config | configs.tasks.segmentation.segmentation_task_config
study_config = study_config | configs.default.default_config

study_config['performance.compile'] = False
study_config['experiment.logging.also_eval_on_train'] = False

study_config['model.normalization'] = "group"

study = Study(study_config)

exp = EXP_OctreeNCA3D().createExperiment(study_config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                            })
study.add_experiment(exp)

study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])

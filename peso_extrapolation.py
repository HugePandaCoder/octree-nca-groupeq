from src.datasets.Dataset_PESO import Dataset_PESO
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA, EXP_OctreeNCA2D_extrapolation
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis


ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"
print("Study Path:", ProjectConfiguration.STUDY_PATH)

study_config = {
        'experiment.name': r'peso_extrapolation_vitca_1_test',
        'experiment.description': "OctreeNCAExtrapolation",
        'experiment.data_split': [0.7, 0, 0.3],
        'experiment.save_interval': 50,
        'experiment.device': "cuda:0",

        'experiment.logging.also_eval_on_train': False,
        'experiment.logging.track_gradient_norm': True,
        'experiment.logging.evaluate_interval': 2001,

        'experiment.dataset.img_path': r"/local/scratch/PESO/peso_training",
        'experiment.dataset.label_path': r"/local/scratch/PESO/peso_training",
        'experiment.dataset.keep_original_scale': True,
        'experiment.dataset.rescale': True,
        'experiment.dataset.input_size': [320, 320],
        'experiment.dataset.img_level': 1,
        

        'experiment.task': "extrapolation",
        'experiment.task.margin': 10, #remove 10 pixels from each border


        'performance.compile': False,
        'performance.data_parallel': False,
        'performance.num_workers': 8,
        'performance.unlock_CPU': True,
        #'performance.inplace_operations': True,


        'trainer.optimizer': "torch.optim.AdamW",
        'trainer.optimizer.lr': 1e-3,
        'trainer.lr_scheduler': "torch.optim.lr_scheduler.CosineAnnealingLR",
        'trainer.lr_scheduler.T_max': 2000,
        'trainer.update_lr_per_epoch': True,
        'trainer.losses': ["torch.nn.L1Loss"],
        'trainer.loss_weights': [1e2],
        'trainer.normalize_gradients': "layerwise", # all, layerwise, none

        'trainer.n_epochs': 2000,
        'trainer.num_steps_per_epoch': 200,
        'trainer.batch_size': 3,
        'trainer.batch_duplication': 1,

        'trainer.find_best_model_on': None,
        'trainer.always_eval_in_last_epochs': None,

        'trainer.ema': True,
        'trainer.ema.decay': 0.999,
        'trainer.ema.update_per': "epoch",

        'trainer.datagen.batchgenerators': True,
        'trainer.datagen.augmentations': True,
        'trainer.datagen.difficulty_weighted_sampling': False,

        'trainer.gradient_accumulation': False,
        'trainer.train_quality_control': False, #or "NQM" or "MSE"



        'model.channel_n': 16,
        'model.fire_rate': 0.5,
        'model.input_channels': 3,
        'model.output_channels': 3,
        'model.kernel_size': [3, 3, 3, 3, 3],
        'model.hidden_size': 64,
        'model.batchnorm_track_running_stats': False,

        'model.train.patch_sizes': [None] * 5,
        'model.train.loss_weighted_patching': False,

        'model.octree.res_and_steps': [[[320, 320], 20], [[160, 160], 20], [[80, 80], 20], [[40, 40], 20], [[20,20], 40]],
        'model.octree.separate_models': True,
        
        'model.vitca': True,
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

###### Define specific model setups here and save them in list ######

study.add_experiment(EXP_OctreeNCA2D_extrapolation().createExperiment(study_config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': study_config['experiment.dataset.patches_path'],
                                                            'patch_size': study_config['experiment.dataset.input_size'],
                                                            'path': study_config['experiment.dataset.img_path'],
                                                            'img_level': study_config['experiment.dataset.img_level']
                                                      }))


study.run_experiments()
study.eval_experiments()
#figure = octree_vis.visualize(study.experiments[0])
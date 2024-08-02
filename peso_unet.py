from src.datasets.Dataset_PESO import Dataset_PESO
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA, EXP_UNet2D
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis


ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"
print(ProjectConfiguration.STUDY_PATH)

study_config = {
    'img_path': r"/local/scratch/PESO/peso_training",
    'label_path': r"/local/scratch/PESO/peso_training",
        'name': r'peso_unet_3',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr_gamma': 0.9999**8,
        'lr': 0.0016,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 50,
        'evaluate_interval': 50,
        'n_epoch': 2000,
        # Model
        'input_channels': 3,
        'output_channels': 1,
        # Data
        'input_size': [(320, 320)], # (320, 320, 24) -> (160, 160, 12) -> (80, 80, 12) -> (40, 40, 12) -> (20, 20, 12)
        
        'data_split': [0.7, 0, 0.3],
        'keep_original_scale': True,
        'rescale': True,
        
        ### TEMP
        'gradient_accumulation': False,
        'train_quality_control': False, #or "NQM" or "MSE"

        'compile': False,
        'data_parallel': False,
        'batch_size': 3,
        'batch_duplication': 1,
        'num_workers': 8,
        'update_lr_per_epoch': True, # is false by default
         # TODO batch duplication per level could be helpful as the levels with a patchsize are much more stochastic than others.
         # Alternativly, train for more epochs and slower weight decay or iterate through all epochs (deterministically, no random sampling of patches)
        'also_eval_on_train': False,
        'num_steps_per_epoch': 200, #default is None
        'train_data_augmentations': False,
        'track_gradient_norm': True,
        'batchgenerators': True, 
        'loss_weighted_patching': False,# default false, train on the patch that has the highest loss in the previous epoch
        # TODO 'lambda_dice_loss'
        # TODO maybe diffulty weighted sampling
        # TODO more data augmentations
        # TODO different weight initializations
        # TODO maybe mask CE loss on correctly segmented areas
        # TODO try adam params (0.5, 0.5)
        'difficulty_weighted_sampling': False, #default is False. Difficulty is evaluated at every 'evaluate_interval' epoch. Also, 'also_eval_on_train' _must_ be True

        'optimizer': "Adam",# default is "Adam"
        'sgd_momentum': 0.99,
        'sgd_nesterov': True,

        'scheduler': "exponential",#default is exponential
        'polynomial_scheduler_power': 1.8,

        'find_best_model_on': None, # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
        'always_eval_in_last_epochs': None, #default is None

        'batchnorm_track_running_stats': False, #default is False

        'apply_ema': True,
        'ema_decay': 0.999,
        'ema_update_per': "epoch",

        
        #smaller initial learning rate
        #other implementation
        #larger patch size (maybe resize)
}

study = Study(study_config)

###### Define specific model setups here and save them in list ######

study.add_experiment(EXP_UNet2D().createExperiment(study_config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': r"/local/scratch/clmn1/data/PESO_patches/",
                                                            'patch_size': study_config['input_size'][0],
                                                            'path': study_config['img_path'],
                                                            'img_level': 1
                                                      }))


study.run_experiments()
study.eval_experiments()
figure = octree_vis.visualize(study.experiments[0])
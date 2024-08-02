from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis


ProjectConfiguration.STUDY_PATH = r"/local/scratch/clmn1/octree_study/"
print(ProjectConfiguration.STUDY_PATH)

study_config = {
    'img_path': r"/local/scratch/AGGC/AGGC2022_train/Subset1_Train_image/",
    'label_path': r"/local/scratch/clmn1/data/AGGC3/AGGC2022_train/Subset1_Train_annotations/",
        'name': r'aggc_23',
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr_gamma': 0.9999**8,
        'lr': 0.0016,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 2,
        'evaluate_interval': 1,
        'n_epoch': 2000,
        # Model
        'input_channels': 3,
        'output_channels': 1,
        'hidden_size': 64,
        'train_model':1,
        'channel_n': 16,
        'kernel_size': [3, 3, 3, 3, 7],
        # Data
        'input_size': [(320, 320)], # (320, 320, 24) -> (160, 160, 12) -> (80, 80, 12) -> (40, 40, 12) -> (20, 20, 12)
        
        'data_split': [0.98, 0, 0.02],
        'keep_original_scale': True,
        'rescale': True,
        # Octree - specific
        'octree_res_and_steps': [((320,320), 20), ((160, 160), 20), ((80, 80), 20), ((40, 40), 20), ((20,20), 40)],
        'separate_models': True,
        # (160, 160, 12) <- (160, 160, 12) <- (80, 80, 12) <- (40, 40, 12) <- (20, 20, 12)
        'patch_sizes':[None] * 5,
        #'patch_sizes': [None] *5,
        ### TEMP
        'gradient_accumulation': False,
        'train_quality_control': False, #or "NQM" or "MSE"

        'compile': True,
        'data_parallel': False,
        'batch_size': 6,
        'batch_duplication': 1,
        'num_workers': 8,
        'update_lr_per_epoch': True, # is false by default
         # TODO batch duplication per level could be helpful as the levels with a patchsize are much more stochastic than others.
         # Alternativly, train for more epochs and slower weight decay or iterate through all epochs (deterministically, no random sampling of patches)
        'also_eval_on_train': True,
        'num_steps_per_epoch': 20, #default is None
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
}

study = Study(study_config)

###### Define specific model setups here and save them in list ######

study.add_experiment(EXP_OctreeNCA().createExperiment(study_config, detail_config={}, 
                                                      dataset_class=Dataset_AGGC, dataset_args={
                                                          'patches_path': r"/local/scratch/clmn1/data/AGGC3/AGGC2022_train/patches/",
                                                          'label_path': study_config["label_path"],
                                                          'img_path': study_config["img_path"],
                                                          'patch_size': (320, 320),
                                                          'merge_all_classes': True,
                                                      }))


study.run_experiments()
study.eval_experiments()
figure = octree_vis.visualize(study.experiments[0])
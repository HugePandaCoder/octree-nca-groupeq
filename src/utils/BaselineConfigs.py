import torch
from src.agents.Agent_M3DNCA_GradAccum import M3DNCAAgentGradientAccum
from src.agents.Agent_M3DNCA_superres import M3DNCAAgent_superres
from src.agents.Agent_M3D_NCA import Agent_M3D_NCA
from src.agents.Agent_MedNCA_extrapolation import MedNCAAgent_extrapolation
from src.losses.WeightedLosses import WeightedLosses
from src.models.UNetWrapper2D import UNetWrapper2D
from src.models.UNetWrapper3D import UNetWrapper3D
from src.models.Model_OctreeNCA_3d_patching import OctreeNCA3DPatch
from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2
from src.utils.ExperimentWrapper import ExperimentWrapper
from src.utils.Experiment import merge_config
import numpy as np
from ..losses.LossFunctions import DiceBCELoss
from torch.utils.data import Dataset

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from unet import UNet2D
from src.agents.Agent_UNet import UNetAgent
    
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss

class EXP_M3DNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'M3DNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 16,        # Number of CA state channels
            'inference_steps': 20,
            'cell_fire_rate': 0.5,
            'batch_size': 4,
            'hidden_size': 64,
            'train_model':3,
            # Data
            'scale_factor': 4,
            'kernel_size': 7,
            'levels': 2,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D()
        model = M3DNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], kernel_size=config['kernel_size'], input_channels=config['input_channels'], levels=config['levels'], scale_factor=config['scale_factor'], steps=config['inference_steps'])
        agent = M3DNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
    
from src.models.Model_MedNCA import MedNCA
from src.agents.Agent_MedNCA_Simple  import MedNCAAgent

class EXP_MEDNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'MEDNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'betas': (0.9, 0.99),
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320) ,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        model = MedNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], steps=config['inference_steps'])
        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
  
from src.models.Model_BasicNCA import BasicNCA

class EXP_BasicNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'MEDNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'betas': (0.9, 0.99),
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320) ,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        model = BasicNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], steps=config['inference_steps'])
        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
    

from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
from src.models.Model_OctreeNCA import OctreeNCA
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
class EXP_OctreeNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = {
            'description': 'OctreeNCA',#OctreeNCA
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'betas': (0.9, 0.99),
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320) ,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset_class is None:
            assert False, "Dataset is None"
        model = OctreeNCA2DPatch2(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], 
                                    output_channels=config['output_channels'], steps=config['inference_steps'],
                        octree_res_and_steps=config['octree_res_and_steps'], separate_models=config['separate_models'],
                        compile=config['compile'], patch_sizes=config['patch_sizes'], kernel_size=config['kernel_size'],
                        loss_weighted_patching=config['loss_weighted_patching'], track_running_stats=config['batchnorm_track_running_stats'])

        #model = OctreeNCAV2(config['channel_n'], config['cell_fire_rate'], 
        #                  device=config['device'], hidden_size=config['hidden_size'], 
        #                  input_channels=config['input_channels'], output_channels=config['output_channels'], 
        #                  steps=config['inference_steps'], octree_res_and_steps=config['octree_res_and_steps'], 
        #                  separate_models=config['separate_models'], compile=config['compile'], kernel_size=config['kernel_size'])
        
        assert config['batchnorm_track_running_stats'] == False
        assert config['gradient_accumulation'] == False
        assert config['train_quality_control'] == False

        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    

class EXP_OctreeNCA2D_extrapolation(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):

        config = study_config
        if dataset_class is None:
            assert False, "Dataset is None"
        model = OctreeNCA2DPatch2(config)
        
        assert config['model.batchnorm_track_running_stats'] == False
        assert config['trainer.gradient_accumulation'] == False
        assert config['trainer.train_quality_control'] == False

        assert config['experiment.task'] == 'extrapolation'
        agent = MedNCAAgent_extrapolation(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)
    
from src.models.Model_OctreeNCA_3D import OctreeNCA3D
class EXP_OctreeNCA3D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = {
            'description': 'OctreeNCA3D',#OctreeNCA
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'kernel_size': 3,
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320),

            #miscellaneous
            'batchnorm_track_running_stats': False,
            'gradient_accumulation': False, #this does not work currently!
            'track_gradient_norm': True, #default is False, but this is just tracking
            'train_quality_control': False, #False or "NQM" or "MSE"
            'inplace_relu': False,

            #EMA
            'apply_ema': False,
            'ema_decay': 0.999,
            'ema_update_per': 'epoch', # 'epoch' or 'batch'

            #instead of EMA, you could find the best model and save it
            'find_best_model_on': None, # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
            'always_eval_in_last_epochs': None,
            

            # Optimizer
            'optimizer': "Adam",# default is "Adam"
            'sgd_momentum': 0.99,
            'sgd_nesterov': True,
            'betas': (0.9, 0.99),

            # LR - Scheduler
            'scheduler': "exponential",#default is exponential
            'lr': 16e-4,
            'polynomial_scheduler_power': 1.8,
            'lr_gamma': 0.9999**8,

            # Data loading
            'num_workers': 4,
            'batchgenerators': True,

        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset_class is None:
            assert False, "Dataset is None"

        if 'patch_sizes' in config:
            model = OctreeNCA3DPatch2(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], 
                                      output_channels=config['output_channels'], steps=config['inference_steps'],
                            octree_res_and_steps=config['octree_res_and_steps'], separate_models=config['separate_models'],
                            compile=config['compile'], patch_sizes=config['patch_sizes'], kernel_size=config['kernel_size'],
                            loss_weighted_patching=config['loss_weighted_patching'], track_running_stats=config['batchnorm_track_running_stats'],
                            inplace_relu=config['inplace_relu'])
        else:
            assert False, "deprecated"
            model = OctreeNCA3D(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], 
                                output_channels=config['output_channels'], steps=config['inference_steps'],
                                octree_res_and_steps=config['octree_res_and_steps'], separate_models=config['separate_models'],
                                compile=config['compile'], kernel_size=config['kernel_size'])
            
        if 'gradient_accumulation' in config and config['gradient_accumulation']:
            agent = M3DNCAAgentGradientAccum(model)
        else:
            agent = M3DNCAAgent(model)
            #agent = Agent_M3D_NCA(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

class EXP_OctreeNCA3D_superres(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = study_config

        if dataset_class is None:
            assert False, "Dataset is None"


        model = OctreeNCA3DPatch2(config)
            
        if 'gradient_accumulation' in config and config['gradient_accumulation']:
            assert False, "not implemented"
            agent = M3DNCAAgentGradientAccum(model)
        else:
            agent = M3DNCAAgent_superres(model)
        loss_function = WeightedLosses(config)

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)




from unet import UNet3D
class EXP_UNet3D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = {
            'description': 'UNet3D',#OctreeNCA
            'batch_duplication': 1,
            # Model
            'batch_size': 12,
            # Data
            'input_size': (320,320),

            #miscellaneous
            'gradient_accumulation': False, #this does not work currently!
            'track_gradient_norm': True, #default is False, but this is just tracking
            'train_quality_control': False, #False or "NQM" or "MSE"
            'inplace_relu': False,

            #EMA
            'apply_ema': False,
            'ema_decay': 0.999,
            'ema_update_per': 'epoch', # 'epoch' or 'batch'

            #instead of EMA, you could find the best model and save it
            'find_best_model_on': None, # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
            'always_eval_in_last_epochs': None,
            

            # Optimizer
            'optimizer': "Adam",# default is "Adam"
            'sgd_momentum': 0.99,
            'sgd_nesterov': True,
            'betas': (0.9, 0.99),

            # LR - Scheduler
            'scheduler': "exponential",#default is exponential
            'lr': 16e-4,
            'polynomial_scheduler_power': 1.8,
            'lr_gamma': 0.9999**8,

            # Data loading
            'num_workers': 4,
            'batchgenerators': True,

        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset_class is None:
            assert False, "Dataset is None"

        model = UNet3D(in_channels=config['input_channels'], out_classes=config['output_channels'], padding=1)
        model = UNetWrapper3D(model)
        if config['compile']:
            model.compile()
            
        assert not ('gradient_accumulation' in config and config['gradient_accumulation'])
        agent = M3DNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)

class EXP_UNet2D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset_class = None, dataset_args = {}):
        config = {
            'description': 'UNet2D',
            'batch_duplication': 1,
            # Model
            'batch_size': 12,
            # Data
            'input_size': (320,320),

            #miscellaneous
            'gradient_accumulation': False, #this does not work currently!
            'track_gradient_norm': True, #default is False, but this is just tracking
            'train_quality_control': False, #False or "NQM" or "MSE"
            'inplace_relu': False,

            #EMA
            'apply_ema': False,
            'ema_decay': 0.999,
            'ema_update_per': 'epoch', # 'epoch' or 'batch'

            #instead of EMA, you could find the best model and save it
            'find_best_model_on': None, # default is None. Can be 'train', 'val' or 'test' whereas 'test' is not recommended
            'always_eval_in_last_epochs': None,
            

            # Optimizer
            'optimizer': "Adam",# default is "Adam"
            'sgd_momentum': 0.99,
            'sgd_nesterov': True,
            'betas': (0.9, 0.99),

            # LR - Scheduler
            'scheduler': "exponential",#default is exponential
            'lr': 16e-4,
            'polynomial_scheduler_power': 1.8,
            'lr_gamma': 0.9999**8,

            # Data loading
            'num_workers': 4,
            'batchgenerators': True,

        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset_class is None:
            assert False, "Dataset is None"
        
        assert config['batchnorm_track_running_stats'] == False
        assert config['gradient_accumulation'] == False
        assert config['train_quality_control'] == False

        model = UNet2D(in_channels=config['input_channels'], out_classes=config['output_channels'], padding=1)
        model = UNetWrapper2D(model)

        if config['compile']:
            model.compile()
        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset_class, dataset_args, loss_function)



    


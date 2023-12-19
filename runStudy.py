#%%
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss

from src.utils.Study import Study

from src.utils.ProjectConfiguration import ProjectConfiguration


from src.utils.BaselineConfigs import EXP_UNet2D, EXP_M3DNCA, EXP_TransUNet, EXP_MEDNCA

###### Define basic configuration here ######

print(ProjectConfiguration.STUDY_PATH)

study_config = {
    #'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
    #'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
    'img_path': r"/home/jkalkhof_locale/Documents/Data/Dataset_BUSI_with_GT/image/",
    'label_path': r"/home/jkalkhof_locale/Documents/Data/Dataset_BUSI_with_GT/label/",
    'name': r'TESTSTUDY_USBreast_17',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 10,
    'evaluate_interval': 5,
    'n_epoch': 10000,
    # Model
    'input_channels': 1,
    'output_channels': 1,
    # Data
    #'input_size': [(320, 320, 32)],
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': False,
    'rescale': True,
}

study = Study(study_config)

from src.datasets.png_seg_Dataset import png_seg_Dataset
dataset = png_seg_Dataset(buffer=True)

###### Define specific model setups here and save them in list ######

study.add_experiment(EXP_MEDNCA().createExperiment(study_config, detail_config={'input_size':(320, 320)}, dataset=dataset)) #detail_config={'input_size':(380,380)},  [(80, 80), (320, 320)]
#study.add_experiment(EXP_M3DNCA().createExperiment(study_config))
#study.add_experiment(EXP_UNet2D().createExperiment(study_config))
#study.add_experiment(EXP_TransUNet().createExperiment(study_config))

###### Define order of models, e.g. run parallel or run one after one ######







###### Also create automatic plots if it is a study ######







###### Run all experiments ######
study.run_experiments()
#study.eval_experiments()
# %%

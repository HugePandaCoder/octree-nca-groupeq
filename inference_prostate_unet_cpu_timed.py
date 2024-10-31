import time
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2
from src.models.UNetWrapper3D import UNetWrapper3D
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet3D
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis, torch, os, json, openslide, math
import einops
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
import numpy as np
import torch.nn.functional as F
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

model_path = "/local/scratch/clmn1/octree_study_new/Experiments/prostate_unet_UNetSegmentation/"


with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_UNet3D().createExperiment(config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})

model: UNetWrapper3D = exp.model
assert isinstance(model, UNetWrapper3D)
model = model.eval().cpu()

dataset_test = exp.datasets['test']

sample = dataset_test[0]['image']
sample = torch.from_numpy(sample).float()
sample = einops.rearrange(sample, "h w d -> 1 h w d 1")



def measure_time_and_print():
    print("about to start inference")
    start = time.time()
    model(sample)
    end = time.time()
    print("Inference time:", end-start)

measure_time_and_print()
measure_time_and_print()
measure_time_and_print()
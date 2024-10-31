import time
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2
from src.utils.BaselineConfigs import EXP_OctreeNCA3D
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

model_path = "/local/scratch/clmn1/octree_study_new/Experiments/prostatefAbl_none_10_1.0_16_OctreeNCASegmentation/"
#model_path = "/local/scratch/clmn1/octree_study_new/Experiments/prostate_m3d_fast_M3dSegmentation/"


with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_OctreeNCA3D().createExperiment(config, detail_config={}, dataset_class=Dataset_NiiGz_3D, dataset_args={})

model: OctreeNCA3DPatch2 = exp.model
assert isinstance(model, OctreeNCA3DPatch2)
model = model.eval().cpu()
model.device = "cpu"
for backbone in model.backbone_ncas:
    backbone.device = "cpu"

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
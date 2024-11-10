from src.datasets.Dataset_PESO import Dataset_PESO
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
import time
import nca_cuda
torch.set_grad_enabled(False)

model_path = "/local/scratch/clmn1/octree_study_new/Experiments/pesoLargeGroupNorm_OctreeNCA2DSegmentation"
model_path = "/local/scratch/clmn1/octree_study_new/Experiments/pesofAbl_none_5_1.0_16_3_1.0_0.99_OctreeNCA2DSegmentation"
model_path = "/local/scratch/clmn1/octree_study_new/Experiments/pesofFixAbl_none_10_1.0_16_3_1.0_0.99_OctreeNCA2DSegmentation"
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_OctreeNCA().createExperiment(config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.patches_path']),
                                                            'patch_size': config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.img_path']),
                                                            'img_level': config['experiment.dataset.img_level']
                                                      })

model: OctreeNCA2DPatch2 = exp.model
assert isinstance(model, OctreeNCA2DPatch2)
model.eval()

def remove_names(x: torch.Tensor):
    x.names = [None] * len(x.names)
    return x

def align_tensor_to(x: torch.Tensor, target: str):
    if isinstance(target, tuple):
        target_str = ' '.join(target)
    elif isinstance(target, str): 
        if max(map(len, target.split())) != 1:
            #targets are like "BCHW"
            target_str = ' '.join(target)
        else:
            #targets are like "B C H W"
            target_str = target
            target = target.replace(" ", "")


    pattern = f"{' '.join(x.names)} -> {target_str}"
    x = remove_names(x)
    x = einops.rearrange(x, pattern)
    x.names = tuple(target)
    return x

def downscale(x: torch.Tensor, out_size):
    x = align_tensor_to(x, "BCHW")
    remove_names(x)

    out = F.interpolate(x, size=out_size)
    out.names = ('B', 'C', 'H', 'W')
    x.names = ('B', 'C', 'H', 'W')
    return out

def compute_resolutions(x_shape, model):
    upscale_factors = []
    for i in range(len(model.octree_res)-1):
        t = []
        for c in range(2):
            t.append(model.octree_res[i][c]//model.octree_res[i+1][c])
        upscale_factors.append(t)

    new_octree_res = [tuple(x_shape)]
    for i in range(1, len(model.octree_res)):
        downsample_factor = np.array(model.octree_res[i-1]) / np.array(model.octree_res[i])
        new_octree_res.append([math.ceil(new_octree_res[i-1][0] / downsample_factor[0]), 
                                math.ceil(new_octree_res[i-1][1] / downsample_factor[1])])
    return new_octree_res


subject = "14"
pos_x, pos_y = 14400, 24320
pos_x, pos_y = 14400 - 1000, 24320-100
#size = (30*161, 30*161)
#size = (700*16, 710*16)
size = (320, 320)

slide = openslide.open_slide(f"/local/scratch/PESO/peso_training/pds_{subject}_HE.tif")
slide = slide.read_region((int(pos_x * slide.level_downsamples[1]),
                           int(pos_y * slide.level_downsamples[1])), 1, size)
#slide = slide.read_region((int(pos_x * slide.level_downsamples[1]),
#                           int(pos_y * slide.level_downsamples[1])), 1, (16*10, 16*10))
slide = np.array(slide)[:,:,0:3]
slide_cpu = slide

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
slide = slide / 255.0
slide = (slide - mean) / std

slide = slide[None]
slide = torch.from_numpy(slide).float()
slide.names = ('B', 'H', 'W', 'C')

slide = align_tensor_to(slide, "BHWC")
computed_resolutions = compute_resolutions(slide.shape[1:3], model)
print(computed_resolutions)

seed = torch.zeros(1, *computed_resolutions[-1], model.channel_n,
                                dtype=torch.float, device=slide.device,
                                names=('B', 'H', 'W', 'C'))
temp = downscale(slide, computed_resolutions[-1])
temp = align_tensor_to(temp, "BHWC")
remove_names(temp)
remove_names(seed)
seed[:,:,:,:model.input_channels] = temp
#temp.names = ('B', 'H', 'W', 'C')
#seed.names = ('B', 'H', 'W', 'C')

torch.cuda.memory._record_memory_history()
torch.cuda.reset_peak_memory_stats()

start_time = time.time()

slide = einops.rearrange(slide, "B H W C -> B C H W")

out = model.forward(slide)

end_time = time.time()
print(f"Time: {end_time - start_time}")

torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
print(torch.cuda.max_memory_allocated() /1000**2)
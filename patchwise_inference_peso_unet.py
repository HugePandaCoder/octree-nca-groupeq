from src.datasets.Dataset_PESO import Dataset_PESO
from src.utils.Study import Study
from src.utils.ProjectConfiguration import ProjectConfiguration
from src.utils.BaselineConfigs import EXP_OctreeNCA
from src.datasets.Dataset_BCSS_Seg import Dataset_BCSS_Seg
from src.datasets.Dataset_AGGC import Dataset_AGGC
import octree_vis, torch, os, json, openslide, math
import einops
from src.models.Model_OctreeNCAV2 import OctreeNCAV2
import numpy as np, time
import torch.nn.functional as F
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.models.Model_OctreeNCA_2d_patching2 import OctreeNCA2DPatch2
import matplotlib.pyplot as plt
from src.utils.BaselineConfigs import EXP_UNet2D

torch.set_grad_enabled(False)
from src.models.UNetWrapper2D import UNetWrapper2D


model_path = "/local/scratch/clmn1/octree_study_new/Experiments/peso_unet_UNet2DSegmentation/"


with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_UNet2D().createExperiment(config, detail_config={}, 
                                                      dataset_class=Dataset_PESO, dataset_args={
                                                            'patches_path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.patches_path']),
                                                            'patch_size': config['experiment.dataset.input_size'],
                                                            'path': os.path.join(pc.FILER_BASE_PATH, config['experiment.dataset.img_path']),
                                                            'img_level': config['experiment.dataset.img_level']
                                                      })

model: UNetWrapper2D = exp.model
assert isinstance(model, UNetWrapper2D)
model = model.eval()
model_device = torch.device("cuda:0")


subject = "14"
pos_x, pos_y = 5000, 28000
subject, pos_x, pos_y = "28", 5000, 23000
subject, pos_x, pos_y = "41", 5000, 28000
subject, pos_x, pos_y = "22", 5000, 18000
#size = (330*16, 340*16)
size = (850*16, 850*16)
PATCH_SIZE = (180*16, 180*16)

print(size)


slide = openslide.open_slide(f"/local/scratch/PESO/peso_training/pds_{subject}_HE.tif")
slide = slide.read_region((int(pos_x * slide.level_downsamples[1]),
                        int(pos_y * slide.level_downsamples[1])), 1, size)
slide = np.array(slide)[:,:,0:3]
slide_cpu = slide

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
slide = slide / 255.0
slide = (slide - mean) / std

slide = slide[None]
slide = torch.from_numpy(slide).float()
slide = einops.rearrange(slide, 'B H W C -> B C H W')

_, _, H ,W = slide.shape


output = torch.zeros((H,W))
for h in range(0, H, PATCH_SIZE[0]):
    for w in range(0, W, PATCH_SIZE[1]):
        patch = slide[:,:,h:h+PATCH_SIZE[0], w:w+PATCH_SIZE[1]]
        out = model(patch.cuda())['logits'].cpu()
        print(out.shape)
        output[h:h+PATCH_SIZE[0], w:w+PATCH_SIZE[1]] = out[0,:,:,0]


output = output.numpy() > 0

np.save("/local/scratch/clmn1/octree_study_new/qualitative/patchwise_inference_peso_unet.npy", output)






import imageio, json, os, torch, einops, math, tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import time

from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
from src.utils.BaselineConfigs import EXP_OctreeNCA3D

torch.set_grad_enabled(False)

from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2


model_path = "/local/scratch/clmn1/octree_study_new/Experiments/cholec_M3dSegmentation/"
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f) 

exp = EXP_OctreeNCA3D().createExperiment(config, detail_config={}, 
                                        dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                        })

model: OctreeNCA3DPatch2 = exp.model
assert isinstance(model, OctreeNCA3DPatch2)
model.eval()

def downscale(x: torch.Tensor, out_size):
    x = model.align_tensor_to(x, "BCHWD")
    model.remove_names(x)

    out = F.interpolate(x, size=out_size)
    out.names = ('B', 'C', 'H', 'W', 'D')
    x.names = ('B', 'C', 'H', 'W', 'D')
    return out

video_path = "/local/scratch/Cholec80/cholec80_full_set/videos/video01.mp4"
video_reader = imageio.get_reader(video_path)
n_frames = video_reader.get_meta_data()['duration'] * video_reader.get_meta_data()['fps']

num_seconds = 11
num_frames = int(video_reader.get_meta_data()['fps'] * num_seconds)

num_frames = 220

video = []
for frame in range(num_frames):
    image = video_reader.get_data(frame)
    video.append(image[None, ...])

video = np.concatenate(video, axis=0)
video = einops.rearrange(video, 't h w c ->  h w (t c)')


outstacks = []
for i in range(math.ceil(video.shape[-1] / 500)):
    outstack = cv2.resize(video[..., i*500:(i+1)*500], (424, 240))
    outstacks.append(outstack)
video = np.concatenate(outstacks, axis=-1)
video = einops.rearrange(video, 'h w (t c) -> t h w c', c=3).astype(np.float32)
video /= 255.0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
video -= mean
video /= std
print(video.shape)

video_tensor = torch.from_numpy(einops.rearrange(video, 'D H W C -> 1 H W D C'))
video_tensor.names = ('B', 'H', 'W', 'D', 'C')
computed_resolutions = model.compute_octree_res(video_tensor)
print(computed_resolutions)

seed = torch.zeros(1, *computed_resolutions[-1], model.channel_n,
                                dtype=torch.float, device=model.device, 
                                names=('B', 'H', 'W', 'D', 'C'))
temp = downscale(video_tensor, computed_resolutions[-1])
temp = model.align_tensor_to(temp, "BHWDC")
model.remove_names(temp)
model.remove_names(seed)
seed[:,:,:,:,:model.input_channels] = temp


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.time()
state = model.backbone_ncas[1](seed, steps=model.inference_steps[1], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[0], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[0])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')
state = model.backbone_ncas[0](state, steps=model.inference_steps[0], fire_rate=model.fire_rate)

end = time.time()
print("Time taken: ", end-start)
print(torch.cuda.max_memory_allocated() / 1000**2)
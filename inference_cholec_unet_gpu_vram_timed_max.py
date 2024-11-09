import imageio, json, os, torch, einops, math, tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import time

from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
from src.models.UNetWrapper3D import UNetWrapper3D
from src.utils.BaselineConfigs import EXP_OctreeNCA3D, EXP_UNet3D

torch.set_grad_enabled(False)



model_path = "/local/scratch/clmn1/octree_study_new/Experiments/cholec_unet_UNetSegmentation/"
with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f)

config['trainer.losses.parameters'] = [None] *10

exp = EXP_UNet3D().createExperiment(config, detail_config={}, 
                                            dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                                'patch_size': config['experiment.dataset.patch_size']
                                            })

model: UNetWrapper3D = exp.model
assert isinstance(model, UNetWrapper3D)
model.eval()

print(model)
exit()

model.input_channels = 3
model.output_channels = 5

video_path = "/local/scratch/Cholec80/cholec80_full_set/videos/video01.mp4"
video_reader = imageio.get_reader(video_path)
n_frames = video_reader.get_meta_data()['duration'] * video_reader.get_meta_data()['fps']

num_seconds = 180
num_frames = int(video_reader.get_meta_data()['fps'] * num_seconds)

num_frames = 4512

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
video_tensor = video_tensor

out_seg = torch.zeros(1, 240, 424, video_tensor.shape[3], model.output_channels, names=('B', 'H', 'W', 'D', 'C'))

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.time()

PATCH_SIZE = 64
PADDING = 16
for i in tqdm.tqdm(range(0, video_tensor.shape[3], PATCH_SIZE)):
    write_start_idx = max(i, 0)
    write_end_idx = min(i+PATCH_SIZE, video_tensor.shape[3])
    load_start_idx = max(i-PADDING, 0)
    load_end_idx = min(i+PATCH_SIZE+PADDING, video_tensor.shape[3])

    padding_start = write_start_idx - load_start_idx
    padding_end = load_end_idx - write_end_idx
    temp_state = video_tensor[:,:,:,load_start_idx:load_end_idx,:]

    temp_seg = model(temp_state.cuda())['logits'].cpu()
    out_seg[:,:,:,write_start_idx:write_end_idx,:] = temp_seg[:,:,:,padding_start:temp_seg.shape[3]-padding_end,:]

end = time.time()
print("Time taken: ", end-start)
print(torch.cuda.max_memory_allocated() / 1000**2)
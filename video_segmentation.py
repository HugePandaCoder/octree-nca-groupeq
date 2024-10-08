import imageio, json, os, torch, einops, math, tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from src.datasets.Dataset_CholecSeg_preprocessed import Dataset_CholecSeg_preprocessed
from src.utils.BaselineConfigs import EXP_OctreeNCA3D

torch.set_grad_enabled(False)

video_path = "/local/scratch/Cholec80/cholec80_full_set/videos/video01.mp4"
video_reader = imageio.get_reader(video_path)
n_frames = video_reader.get_meta_data()['duration'] * video_reader.get_meta_data()['fps']

num_seconds = 180

video = []
for frame in range(int(video_reader.get_meta_data()['fps'] * num_seconds)):
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
video.shape


from src.models.Model_OctreeNCA_3d_patching2 import OctreeNCA3DPatch2


model_path = "/local/scratch/clmn1/octree_study/Experiments/cholec_seg_octree_2_OctreeNCA3D"
with open(os.path.join(model_path, "config.dt")) as f:
    config = json.load(f) 

exp = EXP_OctreeNCA3D().createExperiment(config, detail_config={}, 
                                        dataset_class=Dataset_CholecSeg_preprocessed, dataset_args = {
                                            'use_max_sequence_length_in_eval': False
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



state = model.backbone_ncas[3](seed, steps=model.inference_steps[3], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[2], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[2])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')
state = model.backbone_ncas[2](state, steps=model.inference_steps[2], fire_rate=model.fire_rate)

state = einops.rearrange(state, '1 H W D C -> 1 C H W D')
state = torch.nn.Upsample(size=computed_resolutions[1], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[1])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')

new_state = torch.zeros_like(state)
PATCH_SIZE = 300
PADDING = model.inference_steps[1]
for i in range(0, state.shape[3], PATCH_SIZE):
    write_start_idx = max(i, 0)
    write_end_idx = min(i+PATCH_SIZE, state.shape[3])
    load_start_idx = max(i-PADDING, 0)
    load_end_idx = min(i+PATCH_SIZE+PADDING, state.shape[3])
    #write to new_state[:,:,:,write_start_idx:write_end_idx,:]
    #load from state[:,:,:,load_start_idx:load_end_idx,:]

    padding_start = write_start_idx - load_start_idx
    padding_end = load_end_idx - write_end_idx
    #print(f"[{write_start_idx}, {write_end_idx}], [{load_start_idx}, {load_end_idx}], ({padding_start}, {padding_end})")
    temp_state = state[:,:,:,load_start_idx:load_end_idx,:]
    temp = model.backbone_ncas[1](temp_state, steps=model.inference_steps[1], fire_rate=model.fire_rate)
    #temp = torch.zeros_like(temp_state)


    new_state[:,:,:,write_start_idx:write_end_idx,:] = temp[:,:,:,padding_start:temp.shape[3]-padding_end,:]

state = new_state

state = einops.rearrange(state, '1 H W D C -> 1 C H W D').cpu()
state = torch.nn.Upsample(size=computed_resolutions[0], mode='nearest')(state)
temp = F.interpolate(einops.rearrange(video_tensor, "1 h w t c -> 1 c h w t"), size=computed_resolutions[0])
state[0,:model.input_channels,:,:,:] = temp[0]
state = einops.rearrange(state, '1 C H W D -> 1 H W D C')

del video_tensor, video_reader, video

new_state = torch.zeros(1, *computed_resolutions[0], model.output_channels, names=('B', 'H', 'W', 'D', 'C'))

PATCH_SIZE = 200
PADDING = model.inference_steps[0]
for i in tqdm.tqdm(range(0, state.shape[3], PATCH_SIZE)):
    write_start_idx = max(i, 0)
    write_end_idx = min(i+PATCH_SIZE, state.shape[3])
    load_start_idx = max(i-PADDING, 0)
    load_end_idx = min(i+PATCH_SIZE+PADDING, state.shape[3])
    #write to new_state[:,:,:,write_start_idx:write_end_idx,:]
    #load from state[:,:,:,load_start_idx:load_end_idx,:]

    padding_start = write_start_idx - load_start_idx
    padding_end = load_end_idx - write_end_idx
    #print(f"[{write_start_idx}, {write_end_idx}], [{load_start_idx}, {load_end_idx}], ({padding_start}, {padding_end})")
    temp_state = state[:,:,:,load_start_idx:load_end_idx,:]
    temp = model.backbone_ncas[0](temp_state.to(model.device), steps=model.inference_steps[1], fire_rate=model.fire_rate).cpu()
    #temp = torch.zeros_like(temp_state)


    new_state[:,:,:,write_start_idx:write_end_idx,:] = temp[:,:,:,padding_start:temp.shape[3]-padding_end,model.input_channels:model.input_channels+model.output_channels]

state = new_state

segmentation = (state > 0).numpy()
del state

color_dict={
    0: (252, 111, 3), 
    1: (252, 3, 227), 
    2: (205, 214, 34),
    3: (150, 150, 150),
    4: (0, 173, 29),
}
video_reader = imageio.get_reader(video_path)

out = cv2.VideoWriter('output_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (424, 240), True)
for i in range(segmentation.shape[3]):
    video = video_reader.get_data(i) #HWC
    video = cv2.resize(video, (424, 240))

    frame_seg = segmentation[0, :, :, i]
    frame = np.zeros((segmentation.shape[1], segmentation.shape[2], 3), dtype=np.uint8)
    for c in range(frame.shape[-1]):
        frame[frame_seg[..., c], :] = color_dict[c]
    frame = 0.2 * frame + 0.8 * video
    frame = frame.astype(np.uint8)
    out.write(frame)

out.release()
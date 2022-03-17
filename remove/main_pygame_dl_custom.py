import os
import pygame
import torch
import numpy as np
import matplotlib.pyplot as plt

from lib.displayer import displayer
from lib.utils import mat_distance
from lib.CAModel import CAModel
from lib.utils_vis import to_rgb, make_seed

from src.Datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.Datasets.png_Dataset import png_Dataset
from Experiment import Experiment

import cv2

config = {
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': "models/remaster_1.pth",
    'reload': False,
    'device':"cuda:0",
    'n_epoch': 40,
    # Learning rate
    'lr': 2e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [64],
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 4,
    'persistence_chance':0.5,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
}

os.environ['KMP_DUPLICATE_LIB_OK']='True'

eraser_radius = 3
pix_size = 8

CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
TARGET_PADDING = 0  
model_path = "models/remaster_1.pth"
device = torch.device("cpu")

TARGET_SIZE = 64
_map_shape = (TARGET_SIZE + 2*TARGET_PADDING,TARGET_SIZE + 2*TARGET_PADDING)

_rows = np.arange(_map_shape[0]).repeat(_map_shape[1]).reshape([_map_shape[0],_map_shape[1]])
_cols = np.arange(_map_shape[1]).reshape([1,-1]).repeat(_map_shape[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])

_map = make_seed(_map_shape, CHANNEL_N)

# Define Experiment
dataset = Nii_Gz_Dataset(config['input_size'])
exp = Experiment(config, dataset)
exp.set_model_state('test')

# seed
#if True:
#    dataset = Nii_Gz_Dataset(r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train", size = (TARGET_SIZE, TARGET_SIZE))
#else:
#    dataset = png_Dataset("D:\PhD\Datasets\My_TestPets", size = (TARGET_SIZE, TARGET_SIZE))

target_img, target_label = dataset.__getitem__(5)

p = TARGET_PADDING

# Show output
plt.figure(figsize=(4,4))
plt.imshow(target_label[:,:,0:3]*256)
ax = plt.gca()
ax.set_facecolor('xkcd:cloudy blue')
plt.ion()
plt.show()

pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

_map[:, :, 0:4] = pad_target.cpu()
#######

model = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
output = model(torch.from_numpy(_map.reshape([1,_map_shape[0],_map_shape[1],CHANNEL_N]).astype(np.float32)), 1)

disp = displayer(_map_shape, pix_size)

isMouseDown = False
running = True
i = 0
while running:
    i = i+1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                isMouseDown = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                isMouseDown = False

    if isMouseDown:
        try:
            mouse_pos = np.array([int(event.pos[1]/pix_size), int(event.pos[0]/pix_size)])
            should_keep = (mat_distance(_map_pos, mouse_pos)>eraser_radius).reshape([_map_shape[0],_map_shape[1],1])
            output = torch.from_numpy(output.detach().numpy()*should_keep)
        except AttributeError:
            pass
    if i > 32 and i < 64+32:
        output = model(output, 1)

    _map = to_rgb(output.detach().numpy()[0])
    disp.update(_map)

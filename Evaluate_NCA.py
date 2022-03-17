from re import I
import time
import imageio

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.Datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.Datasets.png_Dataset import png_Dataset

from IPython.display import clear_output

from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_alpha, to_rgb, get_living_mask, make_seed, make_circle_masks

from src.LossFunctions.LossFunctions import DiceLoss
import nibabel as nib
import sys
import os

from src.Agents.Agent_NCA import Agent
from Experiment import Experiment, DataSplit

config = {
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': "models/nca_test_c16_cf05_noOsc_DotInit_100.pth",
    'reload': True,
    'device':"cuda:0",
    'n_epoch': 40,
    # Learning rate
    'lr': 2e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [128],
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 1,
    'persistence_chance':0.5,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
}

# Define Experiment
dataset = Nii_Gz_Dataset(config['input_size'])
exp = Experiment(config, dataset)
exp.set_model_state('test')

device = torch.device("cpu")

ca = CAModel(config['channel_n'], config['cell_fire_rate'], device).to(device)
ca.load_state_dict(torch.load(config['model_path']))

diceLoss = DiceLoss(useSigmoid=False)

agent = Agent(ca, config)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

agent.test(dataset, config, diceLoss, steps=config['inference_steps'][0])

exit()

p = config['target_padding']

_, id, slice = dataset.__getname__(0).split('_')
patient_id = id
patient_3d_image = None
patient_3d_label = None
average_loss = 0
patient_count = 0

#def evaluate(x, target, steps, optimizer, scheduler):
#    x = ca(x, steps=steps)
#    loss = F.mse_loss(x[:, :, :, :3], target)
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#    scheduler.step()
#    return x, loss

for x in range(int(np.floor(dataset.__len__()))):
    # Create images
    batch_seed = np.empty([1, config['target_size'] + 2* config['target_padding'], config['target_size'] + 2* config['target_padding'], config['channel_n']])
    batch_target = np.empty([1, config['target_size'] + 2* config['target_padding'], config['target_size'] + 2* config['target_padding'], 3])
    for j in range(1):
        target_img, target_label = dataset.__getitem__((j+x*1))
        pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
        h, w = pad_target.shape[:2]
        pad_target = np.expand_dims(pad_target, axis=0)
        pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)#

        pad_target_label = np.pad(target_label, [(p, p), (p, p), (0, 0)])
        h, w = pad_target_label.shape[:2]
        pad_target_label = np.expand_dims(pad_target_label, axis=0)
        pad_target_label = torch.from_numpy(pad_target_label.astype(np.float32)).to(device)

        seed = make_seed((h, w), config['channel_n'])# pad_target.cpu()#make_seed((h, w), CHANNEL_N)
        seed[:, :, 0:3] = pad_target.cpu()[0,:,:,:]
        batch_seed[j] = seed
        batch_target[j] = pad_target_label.cpu()

    x0 = torch.from_numpy(batch_seed.astype(np.float32)).to(device)
    batch_target = torch.from_numpy(batch_target.astype(np.float32)).to(device)
    x0 = torch.sigmoid(ca(x0, steps=64)) #np.random.randint(64,96)

    _, id, slice = dataset.__getname__(x).split('_')
    if( id != patient_id):
        loss = 1 - diceLoss(patient_3d_image, patient_3d_label, smooth = 0) #diceLoss(x0[:, :, :, :4], batch_target, smooth = 1)
        # Save 
        patient_3d_image = np.round(patient_3d_image.cpu().detach().numpy())
        nib_image = nib.Nifti1Image(patient_3d_image, np.eye(4))
        nib_label = nib.Nifti1Image(patient_3d_label.cpu().detach().numpy(), np.eye(4))
        nib.save(nib_image, os.path.join(config['out_path'], patient_id + "_image.nii.gz"))  
        nib.save(nib_label, os.path.join(config['out_path'], patient_id + "_label.nii.gz"))  
        
        print(patient_id + ", " + str(loss.item()))
        patient_id = id
        patient_3d_image = None
        patient_3d_label = None
        average_loss = average_loss + loss.item()
        patient_count = patient_count + 1

    if patient_3d_image == None:
        patient_3d_image = x0[:, :, :, 3] #:4
        patient_3d_label = batch_target[:, :, :, 0]
    else:
        patient_3d_image = torch.vstack((patient_3d_image, x0[:, :, :, 3])) #:4
        patient_3d_label = torch.vstack((patient_3d_label, batch_target[:, :, :, 0])) #:4

print("Average Dice -> " + str(average_loss/patient_count))


    
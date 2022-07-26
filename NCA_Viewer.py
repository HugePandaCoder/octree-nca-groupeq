import PySimpleGUI as sg
import os
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.png_Dataset import png_Dataset
from src.datasets.Cityscapes_Dataset import Cityscapes_Dataset
from lib.CAModel_learntPerceive import CAModel_learntPerceive
from src.utils.Experiment import Experiment
from lib.CAModel import CAModel
from src.models.Model_BasicNCA import BasicNCA
import torch
import numpy as np
from lib.utils_vis import to_rgb, to_rgb2, make_seed
import cv2
import base64
from PIL import Image
import io
import time
import matplotlib.pyplot as plt
from src.agents.Agent_NCA import Agent
from src.utils.helper import convert_image, visualize_all_channels, get_img_from_fig, encode, visualize_all_channels_fast, visualize_perceptive_range
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass
from src.datasets.Nii_Gz_Dataset_allpass import Nii_Gz_Dataset_allPass
from lib.CAModel_deeper import CAModel_Deeper

config = [{
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"D:\Cityscape\cityscape_smaller\images\train",
    'label_path': r"D:\Cityscape\cityscape_smaller\labels\train",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'M:/Models/TestNCA_normal_Cityscapes2',
    'device':"cpu",
    'n_epoch': 200,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [128],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 10,
    # Model config
    'channel_n': 64,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 8,
    'repeat_factor': 1,
    'input_channels': 3,
    'input_fixed': True,
    'output_channels': 3,
    # Data
    'input_size': (200, 400),
    'data_split': [0.6, 0, 0.4], 
    'pool_chance': 0.5,
    'Persistence': False,
    'unlock_CPU': True,
}]

config = [{
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:/MasterThesis/Datasets/Hippocampus/preprocessed_dataset_train/imagesTr/",
    'label_path': r"M:/MasterThesis/Datasets/Hippocampus/preprocessed_dataset_train/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'M:/Models/TestNCA_normal_full_LP35_reflectPad_learntPerceive',
    'device':"cpu",
    'n_epoch': 200,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [128],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 10,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.99,
    'cell_fire_interval':None,
    'batch_size': 8,
    'repeat_factor': 1,
    'input_channels': 3,
    'input_fixed': True,
    'output_channels': 3,
    # Data
    'input_size': (64, 64),
    'data_split': [0.6, 0, 0.4], 
    'pool_chance': 0.5,
    'Persistence': False,
    'unlock_CPU': True,
}]

speed_levels = [0, 0.025, 0.05, 0.1, 0.2]

# Define Experiment
dataset = Nii_Gz_Dataset_lowPass()
model = CAModel_learntPerceive(config[0]['channel_n'], config[0]['cell_fire_rate'], torch.device('cpu')).to(torch.device('cpu'))
print("PARAMETERS")
print(model.parameters)
agent = Agent(model)
exp = Experiment(config, dataset, model, agent)
exp.temporarly_overwrite_config(config)
exp.set_model_state('test')


file_list_column = [
    [sg.Text("Combined Image:", background_color='#7B7B7B')],
    [sg.Image(key="-IMAGE_COMBINED-")], 
    [sg.Text("Simulated Perceptive Range:", background_color='#7B7B7B')],
    [sg.Image(key="-PERCEPTIVE_RANGE-")],
    [
        sg.Text("Choose Model", background_color='#7B7B7B'),
        sg.In(size=(25, 1), enable_events=True, key="-Model-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Image Folder", background_color='#7B7B7B'),
        sg.In(size=(25, 1), enable_events=True, key="-Images-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Label Folder", background_color='#7B7B7B'),
        sg.In(size=(25, 1), enable_events=True, key="-Labels-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [sg.Combo(['train', 'val', 'test'],default_value='test',key='-DataType-', readonly=True, enable_events=True)],
]

image_viewer_column = [
    [sg.Text("Choose an image from list on left:", background_color='#7B7B7B')],
    [sg.Text(size=(40, 1), key="-TOUT-", background_color='#7B7B7B')],
    [sg.Image(key="-IMAGE-", expand_x=True, expand_y=True, background_color='#7B7B7B')],
    [sg.Button("Play"), sg.Button("Pause"), sg.Text('Current step: '), sg.Text('0', key="-CURRENT_STEP-")],
    [sg.Text("Speed:", background_color='#7B7B7B'),
    sg.Slider(key="-SPEED_SLIDER-",
        range=(0,4),
         default_value=2,
         size=(20,15),
         orientation='horizontal',
         font=('Helvetica', 12), 
         background_color='#7B7B7B'),
    sg.Text("Log:", background_color='#7B7B7B'),
    sg.Slider(key="-DIVIDE_SLIDER-",
        range=(1,10000),
         default_value=1,
         size=(20,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         enable_events=True, 
         background_color='#7B7B7B'),
    sg.Combo(['nipy_spectral', 'coolwarm', 'gnuplot', 'inferno', 'magma', 'viridis'],default_value='nipy_spectral',key='-ColorMap-', readonly=True, enable_events=True)],
]

layout = [
    [
        sg.Column(file_list_column, expand_x=True, expand_y=True, background_color='#5A5A5A'), #7B7B7B
        sg.VSeperator(),
        sg.Column(image_viewer_column,key="-RIGHT_COLUMN-", expand_x=True, expand_y=True, background_color='#5A5A5A'),
    ]
]

fast = True
out_img = None
overlayed_img = []
perceptive_range = []

def init_input(name):
    _, target_img, target_label = dataset.getItemByName(name)
    target_img, target_label = torch.from_numpy(target_img), torch.from_numpy(target_label)

    print(target_img)
    data = 0, torch.unsqueeze(target_img, 0), torch.unsqueeze(target_label, 0)

    data = agent.prepare_data(data, eval=True)
    _, output, target_label = data

    return output, target_label

def combine_images(img, label):
    #print(label[label == 0])
    return label

def update_image():

    if out_img is not None:
        if fast == True:
            fig_drawn = encode(out_img, size = window['-IMAGE-'].get_size())
        else:
            fig_drawn = get_img_from_fig(out_img, size = window['-IMAGE-'].get_size())
        window["-IMAGE-"].update(data=fig_drawn)
    if len(overlayed_img) != 0:
        combine_image_drawn = encode(overlayed_img, size = (400, 400))
        window["-IMAGE_COMBINED-"].update(data=combine_image_drawn)
    if len(perceptive_range) != 0:
        perceptive_image_drawn = encode(np.log(perceptive_range+1)/3, size = (400, 400))
        window["-PERCEPTIVE_RANGE-"].update(data=perceptive_image_drawn)



def update_content(output, label):
    output_vis = torch.Tensor.numpy(output.clone().detach().cpu())
    label_vis = torch.Tensor.numpy(label.clone().detach().cpu())
    global overlayed_img
    overlayed_img = convert_image(output_vis[...,:3], output_vis[...,3:6], label_vis[...,:3], encode_image=False)
    
    global out_img
    if fast == True:
        out_img = visualize_all_channels_fast(output_vis, replace_firstImage=None, labels=label_vis)
    else:
        out_img = visualize_all_channels(output_vis, replace_firstImage=None, divide_by=int(values["-DIVIDE_SLIDER-"]), labels=label_vis, color_map=values['-ColorMap-'], size = window['-IMAGE-'].get_size())#convert_image(output_vis[...,:3], output_vis[...,3:6], label_vis)
    
    global perceptive_range
    print("AAAAAAAAAAAAAAAAA")
    print(perceptive_range.shape)
    perceptive_range = visualize_perceptive_range(perceptive_range, cell_fire_rate=exp.get_from_config('cell_fire_rate'))

    update_image()

def update_ImageList():
    fnames = dataset.getImagePaths()
    window["-FILE LIST-"].update(fnames)
    window["-FILE LIST-"].update(fnames)



output = None
label = None
window = sg.Window("NCA Viewer", layout, resizable=True, finalize=True, background_color='#323232')

window.bind('<Configure>', "Configure")

playActive = False
i = 0

steps_left = 0
step = 0
pause = 0

while True:
    event, values = window.Read(timeout = 1)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Do Step
    if step != pause:
        step = step +1
        window["-CURRENT_STEP-"].update(str(step))
        start = time.time()
        output = model(output, 1)
        update_content(output, label)
        output = output.detach()
        end = time.time()
        time_passed = end-start
        if(time_passed < speed_levels[int(values["-SPEED_SLIDER-"])]):
            time.sleep(speed_levels[int(values["-SPEED_SLIDER-"])] - time_passed) #"-SPEED_SLIDER-"
        window.refresh()
        steps_left = steps_left -1
        print(steps_left)

    if event == "-Images-":
        update_ImageList()
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            step = 0
            window["-CURRENT_STEP-"].update(str(step))
            pause = 0
            filename = ""
            print("TESTTSTTS")
            output, label = init_input(values["-FILE LIST-"][0])
            perceptive_range = np.zeros((output.shape[1],output.shape[2],3))
            print(perceptive_range.shape)
            update_content(output, label)
            window["-TOUT-"].update(filename)
        except:
            pass
    elif event == "Play":
        pause = pause -1
        steps_left = steps_left + 64
    elif event == "Pause":
        pause = step
    elif event == 'Configure':
        if output is not None:
            update_image()
    elif event == "-DIVIDE_SLIDER-" or event == "-ColorMap":
        update_content(output, label)
    elif event== '-DataType-':
        print(values['-DataType-'])
        exp.set_model_state(values['-DataType-'])
        update_ImageList()
    


window.close()
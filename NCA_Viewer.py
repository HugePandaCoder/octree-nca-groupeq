import PySimpleGUI as sg
import os
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
from src.datasets.png_Dataset import png_Dataset
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
from src.utils.helper import convert_image, visualize_all_channels
from src.datasets.Nii_Gz_Dataset_lowpass import Nii_Gz_Dataset_lowPass

config = [{
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:/MasterThesis/Datasets/Hippocampus/preprocessed_dataset_train/imagesTr/",
    'label_path': r"M:/MasterThesis/Datasets/Hippocampus/preprocessed_dataset_train/labelsTr/",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'M:\Models\TestNCA_lowpass_full_filter1010',
    'device':"cpu",
    'n_epoch': 200,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [64],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 10,
    # Model config
    'channel_n': 16,        # Number of CA state channels
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
    'input_size': (64, 64),
    'data_split': [0.6, 0, 0.4], 
    'pool_chance': 0.5,
    'Persistence': False,
}]

speed_levels = [0, 0.025, 0.05, 0.1, 0.2]

# Define Experiment
dataset = Nii_Gz_Dataset_lowPass(filter="lowpass") #_3D(slice=2)
model = BasicNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], torch.device('cpu')).to(torch.device('cpu'))
print("PARAMETERS")
print(model.parameters)
agent = Agent(model)
exp = Experiment(config, dataset, model, agent)
exp.set_model_state('test')


file_list_column = [
    [
        sg.Text("Choose Model"),
        sg.In(size=(25, 1), enable_events=True, key="-Model-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-Images-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Label Folder"),
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
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Button("Play"),
    sg.Text("Speed:"),
    sg.Slider(key="-SPEED_SLIDER-",
        range=(0,4),
         default_value=2,
         size=(20,15),
         orientation='horizontal',
         font=('Helvetica', 12)),
    sg.Text("Log:"),
    sg.Slider(key="-DIVIDE_SLIDER-",
        range=(1,100),
         default_value=1,
         size=(20,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         enable_events=True)],
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

def init_input(name):
    _, target_img, target_label = dataset.getItemByName(name)
    target_img, target_label = torch.from_numpy(target_img), torch.from_numpy(target_label)

    data = 0, torch.unsqueeze(target_img, 0), torch.unsqueeze(target_label, 0)

    data = agent.prepare_data(data, eval=True)
    _, output, target_label = data

    return output, target_label

def combine_images(img, label):
    #print(label[label == 0])
    return label

def update_image(output, label):
    output_vis = torch.Tensor.numpy(output.clone().detach())
    label_vis = torch.Tensor.numpy(label.clone().detach())
    overlayed_img = convert_image(output_vis[...,:3], output_vis[...,3:6], label_vis, encode_image=False)
    out_img = visualize_all_channels(output_vis, replace_firstImage=overlayed_img, divide_by=int(values["-DIVIDE_SLIDER-"]))#convert_image(output_vis[...,:3], output_vis[...,3:6], label_vis)
    window["-IMAGE-"].update(data=out_img)

def update_ImageList():
    fnames = dataset.getImagePaths()
    window["-FILE LIST-"].update(fnames)



output = None
label = None
window = sg.Window("NCA Viewer", layout)

playActive = False
i = 0

steps_left = 0

while True:
    event, values = window.Read(timeout = 1)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Do Step
    if steps_left > 0:
        start = time.time()
        output = model(output, 1)
        update_image(output, label)
        end = time.time()
        time_passed = end-start
        if(time_passed < speed_levels[int(values["-SPEED_SLIDER-"])]):
            time.sleep(speed_levels[int(values["-SPEED_SLIDER-"])] - time_passed) #"-SPEED_SLIDER-"
        window.refresh()
        steps_left = steps_left -1
        print(steps_left)

    if event == "-Images-":
        #folder = values["-Images-"]
        #try:
        #    # Get list of files in folder
        #    file_list = os.listdir(folder)
        #except:
        #    file_list = []

        update_ImageList()
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = ""
            output, label = init_input(values["-FILE LIST-"][0])
            update_image(output, label)
            window["-TOUT-"].update(filename)
        except:
            pass
    elif event == "Play":
        steps_left = steps_left + 64
    elif event == "-DIVIDE_SLIDER-":
        update_image(output, label)
    elif event== '-DataType-':
        print(values['-DataType-'])
        exp.set_model_state(values['-DataType-'])
        update_ImageList()


window.close()


def convert_image_old(img, label=None):
    img_rgb = to_rgb2(img.detach().numpy()[0][..., :3]) #+ label[0:3]
    label = label[:,:,:3]#.astype(np.float32)
    label_pred = to_rgb2(img.detach().numpy()[0][..., 3:6])

    if label is not None:
        sobel_x = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel = sobel_x + sobel_y
        sobel[:,:,2] = sobel[:,:,0]
        sobel[:,:,0] = 0

        sobel = np.abs(sobel)
        img_rgb[img_rgb < 0] = 0
        label_pred[label_pred < 0] = 0

        img_rgb = sobel * 0.7 + img_rgb * 1 + label_pred#combine_images(img, sobel)#sobel + img * 0.7
    img_rgb = img_rgb * 256
    img_rgb[img_rgb > 256] = 256
    img_rgb = cv2.resize(img_rgb, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.imencode(".png", img_rgb)[1].tobytes()
    return img_rgb
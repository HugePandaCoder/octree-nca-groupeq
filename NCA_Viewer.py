import PySimpleGUI as sg
import os
from Nii_Gz_Dataset import Nii_Gz_Dataset
from png_Dataset import png_Dataset
from Experiment import Experiment
from lib.CAModel import CAModel
import torch
import numpy as np
from lib.utils_vis import to_rgb, to_rgb2, make_seed
import cv2
import base64
from PIL import Image
import io
import time
import matplotlib.pyplot as plt

config = {
    'out_path': r"D:\PhD\NCA_Experiments",
    'img_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\imagesTr",
    'label_path': r"M:\MasterThesis\Datasets\Hippocampus\preprocessed_dataset_train\labelsTr",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': "models/results_pers_noAlpha_keepImage3.pth",
    'reload': True,
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

speed_levels = [0, 0.025, 0.05, 0.1, 0.2]

# Define Experiment
dataset = Nii_Gz_Dataset(config['input_size'])
exp = Experiment(config, dataset)
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
         font=('Helvetica', 12))],
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

model = CAModel(config['channel_n'], config['cell_fire_rate'], torch.device('cpu')).to(torch.device('cpu'))
if config['reload'] == True:
    model.load_state_dict(torch.load(config['model_path']))

def init_input(name):
    target_img, target_label = dataset.getitembyname(name)

    #pad_target = np.pad(target_img, [(0, 0), (0, 0), (0, 0)])
    #h, w = pad_target.shape[:2]
    #pad_target = np.expand_dims(pad_target, axis=0)
    #pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(torch.device('cpu'))
    pad_target = torch.from_numpy(target_img.astype(np.float32)).to(torch.device('cpu'))
    _map_shape = config['input_size']

    _map = make_seed(_map_shape, config['channel_n'])
    _map[:, :, 0:3] = pad_target.cpu()

    output = torch.from_numpy(_map.reshape([1,_map_shape[0],_map_shape[1],config['channel_n']]).astype(np.float32))
    #target_output = torch.from_numpy(target_label.astype(np.float32)).to(torch.device('cpu'))

    return output, target_label

def combine_images(img, label):
    print(label[label == 0])
    return label

def convert_image(img, label=None):
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
    #  #[..., np.newaxis] [1].tobytes() 
    return img_rgb

output = None
label = None
window = sg.Window("NCA Viewer", layout)

playActive = False
i = 0
#window["-Images-"].InitialFolder = r"C:\\"

# init
#event, values = window.read()
#window['-Images-'].InitialFolder = config['img_path']

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-Images-":
        folder = values["-Images-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".nii.gz"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-Images-"], values["-FILE LIST-"][0]
            )
            output, label = init_input(values["-FILE LIST-"][0])
            out_img = convert_image(output, label)#cv2.imencode(".png", to_rgb(output.detach().numpy()[0])*256)[1].tobytes() 

            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(data=out_img)
        except:
            pass
    elif event == "Play":
        for i in range(64):
            print(i)
            start = time.time()
            output = model(output, 1)
            out_img = convert_image(output, label)
            window["-IMAGE-"].update(data=out_img)
            end = time.time()
            time_passed = end-start
            if(time_passed < speed_levels[int(values["-SPEED_SLIDER-"])]):
                time.sleep(speed_levels[int(values["-SPEED_SLIDER-"])] - time_passed) #"-SPEED_SLIDER-"
            window.refresh()

window.close()
import gradio as gr
import numpy as np
from scipy import signal
import os
import pathlib
import time 

from src.datasets.png_Dataset import png_Dataset
from src.models.Model_GrowingNCA import GrowingNCA
import torch
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.models.Model_BasicNCA import BasicNCA
from src.agents.Agent_NCA import Agent
from src.utils.Experiment import Experiment

import cv2

### ------------------- LOAD MODEL --------------------- ###

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config = [{
    #'out_path': r"D:/PhD/NCA_Experiments",
    #'img_path': r"C:\Users\John\Desktop\MecLab",
    #'label_path': r"C:\Users\John\Desktop\MecLab",
    #'data_type': '.nii.gz', # .nii.gz, .jpg
    #'model_path': r'M:/Models/NCA3d_growImage10',
    'out_path': r"D:/PhD/NCA_Experiments",
    'img_path': r"C:\Users\John\Desktop\MecLab\mf\start",
    'label_path': r"C:\Users\John\Desktop\MecLab\mf\end",
    'data_type': '.nii.gz', # .nii.gz, .jpg
    'model_path': r'M:/Models/NCA3d_growImage36_mf',
    'device':"cuda:0",
    'n_epoch': 30000,
    # Learning rate
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    'inference_steps': [40],
    # Training config
    'save_interval': 10,
    'evaluate_interval': 10,
    'ood_interval':100,
    # Model config
    'channel_n': 32,        # Number of CA state channels
    'target_padding': 0,    # Number of pixels used to pad the target image border
    'target_size': 64,
    'cell_fire_rate': 0.5,
    'cell_fire_interval':None,
    'batch_size': 1,
    'repeat_factor': 6,
    'input_channels': 4,
    'input_fixed': True,
    'output_channels': 3,
    # Data
    'input_size': (100, 60),
    'data_split': [0.7, 0, 0.3], 
    'pool_chance': 0.5,
    'Persistence': False,
    'unlock_CPU': True,
}#,
#{
#    'n_epoch': 2000,
#    'Persistence': True,
#}
]

# Define Experiment
dataset = png_Dataset()#_lowPass(filter="random")
device = torch.device('cpu')
#model = BasicNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=128).to(device)
model = GrowingNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=128).to(device)

#ca = medcam.inject(ca, output_dir=r"M:\AttentionMapsUnet", save_maps = True)
agent = Agent(model)
exp = Experiment(config, dataset, model, agent)
exp.temporarly_overwrite_config(config)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))


### ------------------ GRADIO UI --------------------------

size = (0, 0)

output = None

pixelImage = None

def prepareImg():
    #img = np.clip(output[0, ..., 4:7].detach().numpy(), 0, 1)
    img = np.clip(output[0, ..., 0:3].detach().numpy(), 0, 1)
    img2 = img.copy()

    img2[..., 0] = img[..., 2]
    img2[..., 1] = img[..., 1]
    img2[..., 2] = img[..., 0]

    #img2[..., :] = [1, 0, 0]

    #print("IMG2, VALUES, BEGIINNNING", np.min(img2), np.max(img2))
    #print(img2.shape)

    pixel_size = 16

    pixel = load_item(os.path.join(r"C:\Users\John\Downloads", "OnePixel.png"), RGBA=False)
    pixel = cv2.resize(pixel, (pixel_size, pixel_size))

    pixel = np.uint8(pixel*255)
    #pixel = cv2.cvtColor(pixel, cv2.COLOR_BGRA2BGR)

    #print("IMAGE SHAPE", img2.shape)
    
    new_out = np.zeros((img2.shape[0]*pixel_size, img2.shape[1]*pixel_size, 3))
    if True:
        global pixelImage
        if pixelImage is None:
            pixelImage = np.zeros((img2.shape[0]*pixel_size, img2.shape[1]*pixel_size, 3))
            for y in range(img2.shape[1]):
                for x in range(img2.shape[0]): 
                    pos_x = x * pixel_size
                    pos_y = y * pixel_size
                    pixelImage[pos_x:pos_x+pixel_size, pos_y:pos_y+pixel_size, :] = pixel[..., :3] 

        #print("PIXEL IMAGE", np.min(pixelImage), np.max(pixelImage))

        trans_img = np.uint8(img2.copy()*255)
        trans_img = cv2.resize(trans_img, (pixelImage.shape[1], pixelImage.shape[0]), interpolation=cv2.INTER_NEAREST)
        trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2HSV)

        colored_pixelImage = np.uint8(pixelImage.copy())
        colored_pixelImage = cv2.cvtColor(colored_pixelImage, cv2.COLOR_RGB2HSV)


        colored_pixelImage[..., 0] = trans_img[..., 0]
        colored_pixelImage[..., 1] = trans_img[..., 1]
        #print("PIXEL IMAGE", np.min(colored_pixelImage[..., 2]), np.max(colored_pixelImage[..., 2]), np.min(trans_img[..., 2]), np.max(trans_img[..., 2]))
        colored_pixelImage[..., 2] = np.uint8((colored_pixelImage[..., 2]/256) * (trans_img[..., 2]/256) * 255)
        #print("PIXEL IMAGE", np.min(colored_pixelImage[..., 2]), np.max(colored_pixelImage[..., 2]))

        colored_pixelImage = cv2.cvtColor(colored_pixelImage, cv2.COLOR_HSV2RGB) / 256
        if False:
            pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)        

            trans_img = np.uint8(img2*255)
            #img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
            #print("IMG2, VALUES, BEFORE", np.min(trans_img), np.max(trans_img))
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2HSV)
            #trans_img[..., 0] = 100
            #trans_img = cv2.cvtColor(trans_img, cv2.COLOR_HSV2RGB)

            #print("IMG2, VALUES", np.min(trans_img), np.max(trans_img))

            for y in range(trans_img.shape[1]):
                for x in range(trans_img.shape[0]):
                    colored_pixel = pixel.copy()
                    #print("COLORED_PIXEL, VALUES", np.min(colored_pixel), np.max(colored_pixel))
                    colored_pixel[..., 0] = trans_img[x, y, 0] #trans_img[x, y, 0]
                    colored_pixel[..., 1] = trans_img[x, y, 1]
                    colored_pixel[..., 2] = np.uint8((trans_img[x, y, 2]/256) * (colored_pixel[..., 2]/256) * 256)
                    #print("MAX", np.max(trans_img[..., 0]))
                    #print(trans_img[x, y, 0], trans_img.shape)

                    #print("TRANS_IMG, VALUES", np.min(trans_img[x, y, 0]), np.max(trans_img[x, y, 0]))
                    #colored_pixel[:, :, 0] = img2[x, y, 1]
                    #print(img2[x, y, 0], img2[x, y, 1], img2[x, y, 2])
                    colored_pixel = cv2.cvtColor(colored_pixel, cv2.COLOR_HSV2RGB)

                    pos_x = x * pixel_size
                    pos_y = y * pixel_size
                    new_out[pos_x:pos_x+pixel_size, pos_y:pos_y+pixel_size, :] = colored_pixel /256 #trans_img[x, y, :]/256 #
                    #print("NEW_OUT, VALUES", np.min(new_out), np.max(new_out))
                    #new_out = new_out/256

    print("STEP")
    return np.array(colored_pixelImage)#.astype(np.double)

def addEmptyChannels(img):
    
    global size
    size = img.shape[0:2]

    output = np.zeros((1, *size, exp.get_from_config('channel_n')))
    output[0, ..., 0] = img[..., 0]
    output[0, ..., 1] = img[..., 1]
    output[0, ..., 2] = img[..., 2]
    output[0, ..., 3] = img[..., 3]
    output = torch.from_numpy(output).to(torch.float)

    return output

def load_item(path, RGBA=True):
    r"""Loads the data of an image of a given path.
        Args:
            path (String): The path to the nib file to be loaded."""
    #print(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #print("AAAAAAAAAAAAAAAAAAAA", img.shape)
    #img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) #[..., np.newaxis] 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if RGBA:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)/256
    else:
        img = img/256
    return img

def resetImage(matrix):
    global size

    matrix = np.ndarray.flatten(matrix).astype(int)
    size = (matrix[0], matrix[1])
    print(size)

    #REMOVE
    size = (100,60)

    global output
    img = np.zeros((*size, 3)) #np.random.rand(*exp.get_from_config('input_size'))
    
    img = load_item(os.path.join(r"C:\Users\John\Desktop\MecLab\mf\start", "mf.png"))
    #img_middle = tuple(int(el/2) for el in img.shape)
    #img[img_middle[0],img_middle[1], :] = 0
    #print("HERE")
    #print(np.unique(img))
    output = addEmptyChannels(img)

    #outputs.change(np.clip(output[0, ..., 3:6].detach().numpy(), 0, 1))
    
    #outputs.value = prepareImg()
    #outputs.update()
    #print(output.shape)

    return img #prepareImg()

def runInference(img):
    global output

    img = load_item(os.path.join(r"C:\Users\John\Desktop\MecLab\mf\start", "mf.png"))
    output = addEmptyChannels(img) #['image']

    for step in range(80):
        output = model(output, 1)
        time.sleep(0.01)
        yield prepareImg()
        #outputs.value = np.clip(output[0, ..., 3:6].detach().numpy(), 0, 1)
        #outputs.update()
    #return np.clip(output[0, ..., 3:6].detach().numpy(), 0, 1)#.astype(np.double)


#demo = gr.Interface(fn=greet, inputs=gr.Image(type="pil"), outputs = gr.Image(), examples=["test.jpg"])
with gr.Blocks(layout="vertical", display={"height": "100vh"}) as demo:
    with gr.Row():
        with gr.Column():
            outputs = gr.Image(interactive=True, type="numpy", label="Output") #, tool="sketch"
        with gr.Column():
            matrix = gr.DataFrame(col_count=2, row_count=1, interactive=True, label="Inference Size", type="numpy", datatype="number", headers=["height", "width"])
            btn2 = gr.Button("Play")
            btn2.click(fn=runInference, inputs=[outputs], outputs=[outputs])
            btn = gr.Button("Reset")
            btn.click(fn=resetImage, inputs=[matrix], outputs=[outputs]) #outputs
    #with gr.Column():

        #outputs = gr.Image(value="MecLab_lowRes.jpg", label="Training Target: 60x60")
        


demo.queue().launch()
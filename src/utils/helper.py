from cProfile import label
import pickle
import json
import cv2
import numpy as np
import seaborn as sns
import bz2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io

def dump_pickle_file(file, path):
    with open(path, 'wb') as output_file:
        pickle.dump(file, output_file)

def load_pickle_file(path):
    with open(path, 'rb') as input_file:
        file = pickle.load(input_file)
    return file

def dump_compressed_pickle_file(file, path):
    with bz2.BZ2File(path, 'w') as output_file:
        pickle.dump(file, output_file)

def load_compressed_pickle_file(path):
    with bz2.BZ2File(path, 'rb') as input_file:
        file = pickle.load(input_file)
    return file
    
def dump_json_file(file, path):
    with open(path, 'w') as output_file:
        json.dump(file, output_file)

def load_json_file(path):
    with open(path, 'r') as input_file:
        file =  json.load(input_file)
    return file

# https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def get_img_from_fig(fig, dpi=400, size = (1700, 1700)):
    buf = io.BytesIO()

    #min_size = min(size)
    size_inch = fig.get_size_inches()
    size_inch = size / size_inch
    print(size_inch)
    print(size)
    dpi = int(min(size_inch))
    print(dpi)
    #print(size_min)
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    #buf = buf[1].tobytes()
    #img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    #buf.close()
    img = buf
    #img = cv2.imdecode(img_arr, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(img.read())
    return img.read()

def visualize_all_channels(img, replace_firstImage = None, divide_by=3, labels = None, color_map="nipy_spectral", size = (1700, 1700)):
    print(size)
    if img.shape[0] == 1:
        img = img[0]
    if labels is not None and labels.shape[0] == 1:
        labels = labels[0]

    #tiles = int(math.ceil(math.sqrt(img.shape[2])))
    #img_all_channels = img.reshape(3,-1)#np.reshape(img, (img.shape[0]*tiles, img.shape[1]*tiles))

    tiles = int(math.ceil(math.sqrt(img.shape[2])))
    img_x = img.shape[0]
    img_y = img.shape[1]

    img_all_channels = np.zeros((img_x*tiles, img_y*tiles))
    for tile_pos in range(img.shape[2]):
        tile = img[:,:,tile_pos]
        x = tile_pos%tiles
        y = int(math.floor(tile_pos/tiles))
        if tile_pos < 3:
            tile = tile
            if False:
                min = np.min(tile)
                max = np.max(tile)
                tile = tile - min
                tile = np.log2(tile)
                min = np.min(tile)
                max = np.max(tile)
                tile = tile-min / (-min+max)

        img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y] = tile

        tile_pos_lab = tile_pos -3
        if labels is not None and labels.shape[2] > tile_pos_lab and tile_pos_lab > 0:
            tile_label = labels[:,:,tile_pos_lab]

            gx_m1, gy_m1 = np.gradient(tile_label)
            tile_label = gy_m1 * gy_m1 + gx_m1 * gx_m1
            tile_label[tile_label != 0.0] = 1
            img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y][tile_label == 1] = 1000
        
    
    #plt.tight_layout()
    fig, axes = plt.subplots(figsize=(20, 10))
    pos = axes.imshow(img_all_channels, norm=colors.SymLogNorm(linthresh=0.3, linscale=0.3,
                                              vmin=-10.0, vmax=10.0), cmap=color_map)#cmap='RdBu', aspect='auto', vmin=-100, vmax=100)
    
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad = 0.05)

    axes.margins(x= 0, y=0)

    fig.colorbar(pos, cax=cax)
    #fig.subplots_adjust(bottom=0, top=1, left=0.1, right=0.9)
    #fig.tight_layout()
    #plt.show()
    #fig.subplots_adjust(top=1) #, bottom=0.05
    
    fig.canvas.draw()
    img_all_channels = get_img_from_fig(fig, size=size)

    return img_all_channels 

def convert_image(img, prediction, label=None, encode_image=True):
    r"""Convert an image plus an optional label into one image that can be dealt with by Pillow and similar to display
        TODO: Write nicely and optmiize output, currently only for displaying intermediate results
        Args:

            """
    img_rgb = img #+ label[0:3]
    img_rgb = img_rgb - np.amin(img_rgb)
    img_rgb = img_rgb * img_rgb #* img_rgb * 3
    img_rgb = img_rgb / np.amax(img_rgb)
    label_pred = prediction

    img_rgb, label, label_pred = [v.squeeze() for v in [img_rgb, label, label_pred]]

    # Overlay Label on Image
    if label is not None:
        sobel_x = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel = sobel_x + sobel_y
        sobel[:,:,2] = sobel[:,:,0]
        sobel[:,:,0] = 0
        sobel = np.abs(sobel)
        img_rgb[img_rgb < 0] = 0
        label_pred[label_pred < 0] = 0

        img_rgb = np.clip((sobel * 0.8 + img_rgb + 0.5 * label_pred), 0, 1)

    if encode_image:
        img_rgb = encode(img_rgb)
    return img_rgb 

def encode(img_rgb):
    #img_rgb = img_rgb * 255
    #img_rgb[img_rgb > 255] = 255
    factor_y = img_rgb.shape[0] / img_rgb.shape[1] 
    img_rgb = cv2.resize(img_rgb, dsize=(1700, int(1700*factor_y)), interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.imencode(".png", img_rgb)[1].tobytes()
    return img_rgb

def saveNiiGz(self, output, label, patient_id, path):
    output = np.round(output.cpu().detach().numpy())
    output[output < 0] = 0
    output[output > 0] = 1
    nib_image = nib.Nifti1Image(output, np.eye(4))
    nib_label = nib.Nifti1Image(label.cpu().detach().numpy(), np.eye(4))
    nib.save(nib_image, os.path.join(path, patient_id + "_image.nii.gz"))  
    nib.save(nib_label, os.path.join(path, patient_id + "_label.nii.gz"))  
    

r"""Plot individual patient scores
    TODO: 
"""
def loss_log_to_image(loss_log):
    sns.scatterplot(data=loss_log, x="id", y="Dice")

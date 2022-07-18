from cProfile import label
import pickle
import json
import cv2
import numpy as np
import seaborn as sns
import bz2
import math

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

def visualize_all_channels(img, replace_firstImage = None, divide_by=3, labels = None):
    if img.shape[0] == 1:
        img = img[0]
    if labels is not None and labels.shape[0] == 1:
        labels = labels[0]

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

        tile_pos_lab = tile_pos -2
        if labels is not None and labels.shape[2] > tile_pos_lab and tile_pos_lab > 0:
            tile_label = labels[:,:,tile_pos_lab]
            print(tile_label.shape)
            print(labels.shape)

            gx_m1, gy_m1 = np.gradient(tile_label)
            #gx_m2, gy_m2 = np.gradient(label[:,:, 1])
            tile_label = gy_m1 * gy_m1 + gx_m1 * gx_m1
            tile_label[tile_label != 0.0] = 1
            img_all_channels[x*img_x:(x+1)*img_x, y*img_y:(y+1)*img_y][tile_label == 1] = 1000
    
    output_log = False

    if output_log:
        print(np.max(img_all_channels))
        print(np.min(img_all_channels))


    img_all_channels_blue = img_all_channels.copy()
    img_all_channels_blue[img_all_channels_blue >= 1] = 0
    img_all_channels_blue[img_all_channels_blue <= 0] = 0
    if output_log:
        print(np.max(img_all_channels_blue))
        print(np.min(img_all_channels_blue))

    img_all_channels_red = img_all_channels.copy()
    img_all_channels_red[img_all_channels_red > 0] = 0

    img_all_channels_green = img_all_channels.copy()
    img_all_channels_green[img_all_channels_green < 0] = 0

    if output_log:
        print("REEED")
        print(np.max(img_all_channels_red))
        print(np.min(img_all_channels_red))
    #img_all_channels_red = img_all_channels_red * -1
    #img_all_channels_red = img_all_channels_red / 20
    img_all_channels_red = img_all_channels_red * -1
    img_all_channels_red = np.log(img_all_channels_red) 
    img_all_channels_red = (img_all_channels_red) / np.log(divide_by)
    if output_log:
        print(np.max(img_all_channels_red))
        print(np.min(img_all_channels_red))

        print("GREEEN")
        print(np.max(img_all_channels_green))
        print(np.min(img_all_channels_green))
    #img_all_channels_green = img_all_channels_green / 20
    img_all_channels_green = np.log(img_all_channels_green) 
    img_all_channels_green = (img_all_channels_green) / np.log(divide_by)
    if output_log:
        print(np.max(img_all_channels_green))
        print(np.min(img_all_channels_green))

        print("TEEESTST")
    
    img_all_channels = np.stack([img_all_channels_blue, img_all_channels_green, img_all_channels_red], axis=2)
    
    #img_all_channels = np.stack([img_all_channels for _ in range(3)], axis=2)

    max = np.max(img_all_channels)    
    min = np.min(img_all_channels)
    #print(max)
    #print(min)
    #img_all_channels = np.log(img_all_channels)
    #img_all_channels = (img_all_channels - min) / (-min+max)

    #print(img_all_channels.shape)
    if replace_firstImage is not None:
        print("YES")
        print(replace_firstImage.shape)
        img_all_channels[0:img_x, 0:img_y, :] = replace_firstImage


    img_all_channels = cv2.resize(img_all_channels, (1024, 1024), interpolation=cv2.INTER_NEAREST)


    return encode(img_all_channels)

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
    img_rgb = img_rgb * 255
    img_rgb[img_rgb > 255] = 255
    img_rgb = cv2.resize(img_rgb, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
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

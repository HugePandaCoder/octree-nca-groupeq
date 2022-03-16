import pickle
import json
import cv2
import numpy as np
from lib.utils_vis import to_rgb, to_rgb2, make_seed

def dump_pickle_file(file, path):
    with open(path, 'wb') as output_file:
        pickle.dump(file, output_file)

def load_pickle_file(path):
    with open(path, 'rb') as input_file:
        file =  pickle.load(input_file)
    return file

def dump_json_file(file, path):
    with open(path, 'w') as output_file:
        json.dump(file, output_file)

def load_json_file(path):
    with open(path, 'r') as input_file:
        file =  json.load(input_file)
    return file

def convert_image(img, label=None, encode_image=True):
    img_rgb = to_rgb2(img[..., :3]) #+ label[0:3]
    label = label[:,:,:3]#.astype(np.float32)
    img_rgb = img_rgb - np.amin(img_rgb)
    img_rgb = img_rgb * img_rgb #* img_rgb * 3
    img_rgb = img_rgb / np.amax(img_rgb)
    label_pred = to_rgb2(img[..., 3:6])

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
    #img_rgb = img_rgb / np.amax(img_rgb)
    if encode_image:
        img_rgb = img_rgb * 255
        img_rgb[img_rgb > 255] = 255
        img_rgb = cv2.resize(img_rgb, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        img_rgb = cv2.imencode(".png", img_rgb)[1].tobytes()
        return img_rgb
    #  #[..., np.newaxis] [1].tobytes() 
    return img_rgb #np.uint8(img_rgb) #.astype(np.uint8)
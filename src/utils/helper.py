import pickle
import json
import cv2
import numpy as np
from lib.utils_vis import to_rgb, to_rgb2, make_seed
import seaborn as sns
import bz2

def dump_pickle_file(file, path):
    with open(path, 'wb') as output_file:
        pickle.dump(file, output_file)

def load_pickle_file(path):
    with open(path, 'rb') as input_file:
        file =  pickle.load(input_file)
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

def convert_image(img, prediction, label=None, encode_image=True):
    r"""Convert an image plus an optional label into one image that can be dealt with by Pillow and similar to display
        TODO: Write nicely and optmiize output, currently only for displaying intermediate results
        Args:

            """
    img_rgb = to_rgb2(img) #+ label[0:3]
    img_rgb = img_rgb - np.amin(img_rgb)
    img_rgb = img_rgb * img_rgb #* img_rgb * 3
    img_rgb = img_rgb / np.amax(img_rgb)
    label_pred = to_rgb2(prediction)

    img_rgb, label, label_pred = [v.squeeze() for v in [img_rgb, label, label_pred]]

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

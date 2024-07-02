import PIL
import openslide.deepzoom
import tifffile
from torch.utils.data import Dataset
from src.datasets.Data_Instance import Data_Container
from src.datasets.Dataset_Base import Dataset_Base
import cv2
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize
from PIL import Image
from os import listdir
from os.path import join
import os
import random

import zarr
import openslide

class Dataset_AGGC(Dataset_Base):
    #https://aggc22.grand-challenge.org/Data/
    def __init__(self) -> None:
        super().__init__()
        self.slice = 2
        self.delivers_channel_axis = True
        self.is_rgb = True

    def getFilesInPath(self, path: str):
        files = os.listdir(path)
        dic = {}
        for f in files:
            dic[f]={}
            dic[f][0] = f
        return dic
    
    def __getitem__(self, idx: str):
        file_path = os.path.join(self.images_path, self.images_list[idx])
        openslide_image = openslide.OpenSlide(file_path)

        pos_x = random.randint(0, openslide_image.level_dimensions[0][0] - self.size[0])
        pos_y = random.randint(0, openslide_image.level_dimensions[0][1] - self.size[1])

        img = openslide_image.read_region((pos_x, pos_y), 0, self.size)

        label = np.zeros((self.size[0], self.size[1], 5), dtype=int)

        for i, seg_class in enumerate(["Stroma", "Normal", "G3", "G4", "G5"]):
            label_path = os.path.join(self.labels_path, self.images_list[idx][:-len(".tiff")], f"{seg_class}_Mask.npy")
            if not os.path.exists(label_path):
                continue
            
            mmapped_lbl = np.memmap(label_path, dtype=bool, mode='r', 
                                    shape=(openslide_image.level_dimensions[0][0], openslide_image.level_dimensions[0][1]))


            label[:,:,i] = mmapped_lbl[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1]]

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = np.array(img, dtype=float)
        img = img[:,:,:3]
        img = img / 255.0
        img = (img - mean) / std



        data_dict = {}
        data_dict['id'] = self.images_list[idx]
        data_dict['image'] = img
        data_dict['label'] = label

        return data_dict
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

class Dataset_BCSS_Seg(Dataset_Base):
    def __init__(self) -> None:
        pass

    def getFilesInPath(self, path: str):
        files = os.listdir(path)
        dic = {}
        for f in files:
            dic[f]={}
            dic[f][0] = f
        return dic
    
    def __getitem__(self, idx: str):
        file_path = os.path.join(self.images_path, self.images_list[idx])
        label_path = os.path.join(self.labels_path, self.labels_list[idx])
        mmapped_image = tifffile.memmap(file_path)
        mmapped_label = tifffile.memmap(label_path)

        pos_x = random.randint(0, mmapped_image.shape[0] - self.size[0])
        pos_y = random.randint(0, mmapped_image.shape[1] - self.size[1])

        img = mmapped_image[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1]]
        lbl = mmapped_label[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1]]

        print(lbl.unique())
        exit()

        data_dict = {}
        data_dict['id'] = self.images_list[idx]
        data_dict['image'] = img
        data_dict['label'] = lbl

        return data_dict
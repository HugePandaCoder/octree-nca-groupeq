from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import nibabel as nib
import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from  src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D

class png_Dataset(Dataset_NiiGz_3D):

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            Args:
                path (String): The path to the nib file to be loaded."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) #[..., np.newaxis]
        img = img/256 #cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)/256
        return img

    def __getitem__(self, idx):
        r"""Standard get item function
            Args:
                idx (int): Id of item to loa
            Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        img = self.data.get_data(key=self.images_list[idx])
        if not img:
            img_name, p_id, img_id = self.images_list[idx]
            label_name, _, _ = self.labels_list[idx]

            #img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, img_name))
            img = self.load_item(os.path.join(self.images_path, img_name))
            label = self.load_item(os.path.join(self.labels_path, img_name))

            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])

        id, img, label = img

        #img = np.zeros(img.shape) 

        #print(img.shape)

        #img_middle = tuple(int(el/2) for el in img.shape)
        #img[img_middle[0],img_middle[1], :] = 0.1

        img = img[...,0:4]

        return (id, img, label)

    #def __init__(self, root_dir, size):
    #    self.root_dir = root_dir
     #   self.images_list = listdir(join(root_dir, "images"))
     #   self.length = [f for f in listdir(join(root_dir, "images")) if isfile(join(join(root_dir, "images"), f))]
     #   self.size = size


    #def __len__(self):
    #    return self.length

    #def __getitem__(self, idx):

        #img = cv2.imread(join(self.root_dir, "images", self.images_list[idx]))
        #img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) #[..., np.newaxis] 
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)/256
        #img[:,:,3] = 1

        #label_path = join(self.root_dir, "trimaps", self.images_list[idx])[:-4] + ".png"
        #label = cv2.imread(label_path)
        #if(label != None):
        #    label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) #[..., np.newaxis] 
        #    label = cv2.cvtColor(label, cv2.COLOR_RGB2RGBA)
        #    label[:,:, 3] = np.clip(np.round(label[:,:, 0]),0,1)

        #    label[:,:, 0] = label[:,:, 0] == 2
        #    label[:,:, 1] = label[:,:, 1] == 1
        #    label[:,:, 2] = label[:,:, 2] == 3

        #return img, label


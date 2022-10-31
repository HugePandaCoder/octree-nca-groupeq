from cmath import nan
import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import nibabel as nib
import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from src.datasets.Data_Instance import Data_Container
from src.datasets.Dataset_Base import Dataset_Base

class Nii_Gz_Dataset(Dataset_Base):
    r""".. WARNING:: Deprecated, lacks functionality of 3D counterpart. Needs to be updated to be useful again."""

    def getFilesInPath(self, path):
        r"""Get files in path ordered by id and slice
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            _, id, slice = f.split("_")
            if id not in dic:
                dic[id] = {}
            dic[id][slice] = f
        return dic

    def __getname__(self, idx):
        r"""Get name of item by id"""
        return self.images_list[idx]

    def __getitem__(self, idx):
        r"""Standard get item function
            Args:
                idx (int): Id of item to loa
            Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        img_id = self.__getname__(idx)
        out = self.data.get_data(key=img_id)
        if out == False:
            img = nib.load(os.path.join(self.images_path, self.images_list[idx])).get_fdata()
            label = nib.load(os.path.join(self.labels_path, self.labels_list[idx])).get_fdata()[..., np.newaxis]
            img, label = self.preprocessing(img, label)
            self.data.set_data(key=img_id, data=(img_id, img, label))

            out = self.data.get_data(key=img_id)
        return out

    def getIdentifier(self, idx):
        r""".. TODO:: Remove redundancy"""
        return self.__getname__(idx)

    def preprocessing(self, img, label):
        r"""Preprocessing of image
            Args:
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """
        
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        #img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)  #np.array(img) #
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img[np.isnan(img)] = 1

        #plt.imshow(img)
        #plt.show()

        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        #label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) #np.squeeze(np.array(label)) #
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)

        label[:,:, 0] = label[:,:, 0] != 0
        
        #label[:,:, 1] = label[:,:, 1] != 0
        label[:,:, 1] = 0

        # Edges as second mask
        #gx, gy = np.gradient(label[:,:, 0])
        
        #label[:, :, 1] = gy * gy + gx * gx
        #label[:, :, 1][label[:, :, 1] != 0.0] = 1
        
        #label[:,:, 1] = 0
        label[:,:, 2] = 0

        return img, label


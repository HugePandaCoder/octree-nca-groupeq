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

class Nii_Gz_Dataset(Dataset):

    def __init__(self): 
        self.size = (64, 64)
        return

    r"""Get files in path ordered by id and slice
        Args:
            path (string): The path which should be worked through
        Returns:
            dic (dictionary): {key:patientID, {key:sliceID, img_slice}
    """
    def getFilesInPath(self, path):
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            _, id, slice = f.split("_")
            if id not in dic:
                dic[id] = {}
            dic[id][slice] = f
        return dic

    r"""TODO"""
    def setPaths(self, images_path, images_list, labels_path, labels_list):
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = len(self.images_list)

    r"""Set size of images
        Args:
            size (int, int): Size of images
    """
    def set_size(self, size):
        self.size = tuple(size)

    r"""Get number of items in dataset"""
    def __len__(self):
        return self.length

    r"""Get name of item by id"""
    def __getname__(self, idx):
        return self.images_list[idx]

    r"""Get item by its name
        Args:
            name (string): Name of item
    """
    def getitembyname(self, name):
        img = nib.load(os.path.join(self.images_path, name)).get_fdata()
        label = nib.load(os.path.join(self.labels_path, name)).get_fdata()[..., np.newaxis]
        return self.processing(img, label)

    r"""Standard get item function
        Args:
            idx (int): Id of item to loa
        Returns:
            img (numpy): Image data
            label (numpy): Label data
    """
    def __getitem__(self, idx):
        img = nib.load(os.path.join(self.images_path, self.images_list[idx])).get_fdata()
        label = nib.load(os.path.join(self.labels_path, self.labels_list[idx])).get_fdata()[..., np.newaxis]
        return self.processing(img, label)

    r"""TODO: Remove redundancy"""
    def getIdentifier(self, idx):
        return self.__getname__(idx)

    r"""Preprocessing of image
        Args:
            img (numpy): Image to preprocess
            label (numpy): Label to preprocess
    """
    def processing(self, img, label):
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)

        label[:,:, 0] = label[:,:, 0] != 0
        label[:,:, 1] = 0
        label[:,:, 2] = 0

        return img, label


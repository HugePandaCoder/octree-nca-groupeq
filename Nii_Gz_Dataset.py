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

    def __init__(self): #size , root_dir, size
        #self.root_dir = root_dir
        #self.images_list = listdir(join(root_dir, "imagesTr"))
        #self.labels_list = listdir(join(root_dir, "labelsTr"))
        #self.length = [f for f in listdir(join(root_dir, "imagesTr")) if isfile(join(join(root_dir, "imagesTr"), f))]
        self.size = (64, 64)
        return

    # Standard Method to replace depending on naming scheme
    # Dictionary with, {key:patientID, {key:sliceID, img_slice}
    def getFilesInPath(self, path):
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            _, id, slice = f.split("_")
            if id not in dic:
                dic[id] = {}
            dic[id][slice] = f
        #print(dic)
        return dic

    def setPaths(self, images_path, images_list, labels_path, labels_list):
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = len(self.images_list)

    def set_size(self, size):
        self.size = tuple(size)

    def __len__(self):
        return self.length

    def __getname__(self, idx):
        return self.images_list[idx]

    def getitembyname(self, name):
        img = nib.load(os.path.join(self.images_path, name)).get_fdata()
        label = nib.load(os.path.join(self.labels_path, name)).get_fdata()[..., np.newaxis]
        return self.processing(img, label)

    def __getitem__(self, idx):
        img = nib.load(os.path.join(self.images_path, self.images_list[idx])).get_fdata()
        label = nib.load(os.path.join(self.labels_path, self.labels_list[idx])).get_fdata()[..., np.newaxis]
        return self.processing(img, label)

    def getIdentifier(self, idx):
        return self.images_list[idx]

    def processing(self, img, label):
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)  #[..., np.newaxis] 
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #img[:,:, 3] = 1

        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) #[..., np.newaxis] 
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)
        #label[:,:, 3] = np.round(label[:,:, 0])

        label[:,:, 0] = label[:,:, 0] != 0
        #label[:,:, 0] = np.round(label[:,:, 0]) == 1
        #label[:,:, 1] = np.round(label[:,:, 1]) == 2
        #label[:,:, 2] = np.round(label[:,:, 2]) == 3
        #label[:,:, 0] = label[:,:, 0] or label[:,:, 1] or label[:,:, 2]
        label[:,:, 1] = 0
        label[:,:, 2] = 0
        #label[:,:, 3] = 1

        #print(img.shape)
        #print(label.shape)

        return img, label


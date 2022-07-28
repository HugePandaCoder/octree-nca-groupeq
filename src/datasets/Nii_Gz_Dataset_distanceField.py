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
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import matplotlib.pyplot as plt
import skfmm

class Nii_Gz_Dataset_DistanceField(Nii_Gz_Dataset):
    r""".. WARNING:: Deprecated, lacks functionality of 3D counterpart. Needs to be updated to be useful again."""


    def preprocessing(self, img, label):
        r"""Preprocessing of image
            Args:
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """
        
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img[np.isnan(img)] = 1

        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)

        label[:,:, 0] = label[:,:, 0] != 0
        label[:,:, 1] = 0
        label[:,:, 2] = 0

        label[:,:, 0] = self.createDistanceField(label[:,:, 0])

        label = cv2.normalize(label, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return img, label

    def createDistanceField(self, label_layer):

        #label_layer = self.distanceField_oneDirection(label_layer, range(label_layer.shape[1]), range(label_layer.shape[0]), [1, 0])

        if False:
            dir = [1, 0]
            for y in range(label_layer.shape[1]):
                for x in range(label_layer.shape[0]):
                    if label_layer[x, y] != 1 and x-dir[0] > 0 and x-dir[0] < label_layer.shape[0] and y-dir[1] > 0 and y-dir[1] < label_layer.shape[1]:
                        label_layer[x, y] = max(0, label_layer[x, y], label_layer[x-dir[0], y-dir[1]]-0.1)

            dir = [-1, 0]
            for y in reversed(range(label_layer.shape[1])):
                for x in reversed(range(label_layer.shape[0])):
                    if label_layer[x, y] != 1 and x-dir[0] > 0 and x-dir[0] < label_layer.shape[0] and y-dir[1] > 0 and y-dir[1] < label_layer.shape[1]:
                        label_layer[x, y] = max(0, label_layer[x, y], label_layer[x-dir[0], y-dir[1]]-0.1)

            dir = [0, 1]
            for y in range(label_layer.shape[1]):
                for x in range(label_layer.shape[0]):
                    if label_layer[x, y] != 1 and x-dir[0] > 0 and x-dir[0] < label_layer.shape[0] and y-dir[1] > 0 and y-dir[1] < label_layer.shape[1]:
                        label_layer[x, y] = max(0, label_layer[x, y], label_layer[x-dir[0], y-dir[1]]-0.1)

            dir = [0, -1]
            for y in reversed(range(label_layer.shape[1])):
                for x in reversed(range(label_layer.shape[0])):
                    if label_layer[x, y] != 1 and x-dir[0] > 0 and x-dir[0] < label_layer.shape[0] and y-dir[1] > 0 and y-dir[1] < label_layer.shape[1]:
                        label_layer[x, y] = max(0, label_layer[x, y], label_layer[x-dir[0], y-dir[1]]-0.1)
        
        # Differentiate the inside / outside region
        #phi = np.int64(np.any(label_layer, axis=1))
        # The array will go from - 1 to 0. Add 0.5(arbitrary) so there 's a 0 contour.
        #phi = np.where(phi, 0, -1) + 0.5

        label_layer = skfmm.distance(label_layer*3, dx=[1, 1])

        #plt.imshow(label_layer)
        #plt.show()
            
        return label_layer

    def distanceField_oneDirection(self, label_layer, range1, range2, dir):
        for y in range1:
            for x in range2:
                if label_layer[x, y] != 1 and x-dir[0] > 0 and x-dir[0] < label_layer.shape[0] and y-dir[1] > 0 and y-dir[1] < label_layer.shape[1]:
                    label_layer[x, y] = max(0, label_layer[x, y], label_layer[x-dir[0], y-dir[1]]-0.1)

        return label_layer


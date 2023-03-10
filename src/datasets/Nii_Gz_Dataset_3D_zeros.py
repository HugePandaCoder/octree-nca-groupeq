from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import nibabel as nib
import torch
import os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import random
import torchio

class Dataset_NiiGz_3D_zeros(Dataset_NiiGz_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            Args:
                path (String): The path to the nib file to be loaded."""
        return np.random.rand(50, 50, 50) #nib.load(path).get_fdata()
from torch.utils.data import Dataset
import cv2
import numpy as np

class Dataset_3D(Dataset):
    r"""Base class to load 3D datasets
        - Not to be used directly!
    """
    def __init__(self, slice=None, resize=True): 
        self.slice = slice
        self.resize = resize

    def set_size(self, size):
        r"""Set size of images
            Args:
                size (int, int): Size of images
        """
        self.size = tuple(size)

    def setPaths(self, images_path, images_list, labels_path, labels_list):
        r"""TODO"""
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = len(self.images_list)
    
    def resize_image(self, img, isLabel):
        r"""TODO REMOVE OR USE"""
        if not isLabel:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        return img
    
    def preprocessing(self, img, isLabel=False):

        if not isLabel:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST) 

        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # TODO: REMOVE
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if isLabel:
            img[...,1] = 0
            img[...,2] = 0
        
        return img
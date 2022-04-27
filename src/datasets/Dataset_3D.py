from torch.utils.data import Dataset
import cv2
import numpy as np

class Dataset_3D(Dataset):
    r"""Base class to load 3D datasets
        .. WARNING:: Not to be used directly!
    """
    def __init__(self, slice=None, resize=True): 
        self.slice = slice
        self.resize = resize

    def set_size(self, size):
        r"""Set size of images. Images will later be rescaled if necessary and not disabled.
            Args:
                size (int, int): Size of images
        """
        self.size = tuple(size)

    def getImagePaths(self):
        r"""Get a list of all images in dataset
            Returns:
                list ([String]): List of images
        """
        return self.images_list

    def getItemByName(self, name):
        r"""Get item by its name
            Args:
                name (String)
            Returns:
                item (tensor): The image tensor
        """
        idx = self.images_list.index(name)
        return self.__getitem__(idx)


    def setPaths(self, images_path, images_list, labels_path, labels_list):
        r"""Set the important image paths
            Args:
                images_path (String): The path to the images
                images_list ([String]): A list of the names of all images
                labels_path (String): The path to the labels
                labels_list ([String]): A list of the names of all labels
            .. TODO:: Refactor
        """
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
        r"""Preprocessing of image slices
            Args:
                img (tensor): the image
                isLabel (boolean): Whether its a mask or an image
            .. warning:: Likely there is a preprocessing problem since performance is worse than the already preprocessed slices. ( I imagine the scaling functionality of the mask is at fault)
        """
        if not isLabel:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        
        # TODO: REMOVE
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if isLabel:
            img[...,1] = 0
            img[...,2] = 0
        
        return img

    def __getitem__():
        r"""Placeholder function for getting a dataset value"""
        return None, None, None
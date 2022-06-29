from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import torch
import os
import numpy as np
from src.datasets.Data_Instance import Data_Container
from scipy import fftpack
from PIL import Image, ImageDraw
import cv2

class Dataset_NiiGz_3D_lowpass(Dataset_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """

    def __init__(self, slice=None, resize=True, filter="lowpass"): 
        self.slice = slice
        self.resize = resize
        self.data = Data_Container()
        self.filter = filter

    def preprocessing(self, img, isLabel=False):
        r"""Preprocessing of image slices
            Args:
                img (tensor): the image
                isLabel (boolean): Whether its a mask or an image
            .. warning:: Likely there is a preprocessing problem since performance is worse than the already preprocessed slices. ( I imagine the scaling functionality of the mask is at fault)
        """
        if not isLabel:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
            if self.filter == "lowpass":
                img = self.lowpass_filter(img)
            else:
                img = self.highpass_filter(img)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        
        # TODO: REMOVE
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        if isLabel:
            img[...,1] = 0
            img[...,2] = 0
        
        return img

    def lowpass_filter(self, image):
        #image = image.numpy()
        fft1 = fftpack.fftshift(fftpack.fft2(image))

        x,y = image.shape[0], image.shape[1]
        #size of circle
        e_x,e_y=10,10
        #create a box 
        bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))
        low_pass=Image.new("L",(image.shape[0],image.shape[1]),color=0)

        draw1=ImageDraw.Draw(low_pass)
        draw1.ellipse(bbox, fill=1)

        low_pass_np=np.array(low_pass)

        #multiply both the images
        filtered=np.multiply(fft1,low_pass_np)

        #inverse fft
        ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
        #ifft2 = np.maximum(0, np.minimum(ifft2, 255))

        #ifft2 = torch.from_numpy(ifft2)

        return ifft2

    def highpass_filter(self, image):
        #image = image.numpy()
        fft1 = fftpack.fftshift(fftpack.fft2(image))

        x,y = image.shape[0], image.shape[1]
        #size of circle
        e_x,e_y=10,10
        #create a box 
        bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))
        low_pass=Image.new("L",(image.shape[0],image.shape[1]),color=0)

        draw1=ImageDraw.Draw(low_pass)
        draw1.ellipse(bbox, fill=1)

        low_pass_np=np.array(low_pass)
        low_pass_np = 1 - low_pass_np

        #multiply both the images
        filtered=np.multiply(fft1,low_pass_np)

        #inverse fft
        ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
        #ifft2 = np.maximum(0, np.minimum(ifft2, 255))

        #ifft2 = torch.from_numpy(ifft2)

        return ifft2

    def getDataShapes():
        return

    def getFilesInPath(self, path):
        r"""Get files in path ordered by id and slice
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = os.listdir(os.path.join(path))
        dic = {}
        for id_f, f in enumerate(dir_files):
            id = f
            # 2D ? 
            if self.slice is not None:
                for slice in range(self.getSlicesOnAxis(os.path.join(path, f), self.slice)):
                    if id not in dic:
                        dic[id] = {}
                    dic[id][slice] = (f, id_f, slice)
            else:
                if id not in dic:
                    dic[id] = {}
                dic[id][0] = f           
        return dic

    def getSlicesOnAxis(self, path, axis):
        return self.load_item(path).shape[axis]

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            Args:
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()

    def __len__(self):
        r"""Get number of items in dataset"""
        return self.length

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

            img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, img_name))
            if self.slice is not None:
                if self.slice == 0:
                    img, label = img[img_id, :, :], label[img_id, :, :]
                elif self.slice == 1:
                    img, label = img[:, img_id, :], label[:, img_id, :]
                elif self.slice == 2:
                    img, label = img[:, :, img_id], label[:, :, img_id]
                img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
            img_id = "_" + str(p_id) + "_" + str(img_id)
            
            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])

        return img

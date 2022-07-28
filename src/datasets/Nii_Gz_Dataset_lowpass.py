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
from scipy import fftpack
from PIL import Image, ImageDraw
from src.datasets.Nii_Gz_Dataset import Nii_Gz_Dataset
import torchio


class Nii_Gz_Dataset_lowPass(Nii_Gz_Dataset):
    r""".. WARNING:: Deprecated, lacks functionality of 3D counterpart. Needs to be updated to be useful again."""

    def __init__(self, filter="lowpass", aug_type=None, e_x = 10, e_y = 10): 
        self.size = (64, 64)
        self.data = Data_Container()
        self.filter = filter
        self.aug_type = aug_type
        torch.manual_seed(42)
        self.e_x = e_x
        self.e_y = e_y

    def lowpass_filter(self, image):
        #image = image.numpy()
        fft1 = fftpack.fftshift(fftpack.fft2(image))

        x,y = image.shape[0], image.shape[1]
        #size of circle
        e_x,e_y=self.e_x, self.e_y
        #create a box 
        bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))
        low_pass=Image.new("L",(image.shape[0],image.shape[1]),color=0)

        draw1=ImageDraw.Draw(low_pass)
        draw1.ellipse(bbox, fill=1)

        low_pass_np=np.array(low_pass)

        #multiply both the images
        filtered=np.multiply(fft1,low_pass_np)

        #imgPIL = Image.fromarray(np.uint8(filtered))
        #imgPIL.save("C:/Users/John/Desktop/testnca.jpg")
        #cv2.imwrite("C:/Users/John/Desktop/testnca.jpg", filtered)

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
        e_x,e_y=self.e_x, self.e_y
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

    def torchio_augmentation(self, img, aug_type=None):
        #print(aug_type)
        if aug_type == None:
            return img
        if aug_type == "random_noise":
            transform = torchio.RandomNoise(mean=100, std=200)
        if aug_type == "random_spike":
            transform = torchio.RandomSpike(num_spikes=7, intensity=(0, 0.5))
        if aug_type == "random_anitrosopy":
            transform = torchio.RandomAnisotropy(downsampling=(3.5, 9))

        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = transform(img)
        img = np.squeeze(img)
        return img

    def preprocessing(self, img, label):
        r"""Preprocessing of image
            Args:
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """
        
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        img = self.torchio_augmentation(img, aug_type=self.aug_type)
        if self.filter == "lowpass":
            img = self.lowpass_filter(img)
        elif self.filter == "random":
            #print(np.random.randint(0, 2))
            if np.random.randint(0, 2) == 1:
                img = self.highpass_filter(img)
            else:
                img = self.lowpass_filter(img)
        else:
            img = self.highpass_filter(img)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img[np.isnan(img)] = 1

        label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)

        label[:,:, 0] = label[:,:, 0] != 0
        label[:,:, 1] = 0
        label[:,:, 2] = 0

        return img, label


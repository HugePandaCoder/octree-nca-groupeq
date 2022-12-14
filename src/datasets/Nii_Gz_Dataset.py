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
import random
import torchio

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

            #Random flip data - wrong examples
            if False:
                if random.uniform(0, 1) < 0.7:
                    print("ROLLLL; REMOVE THIS!!!!!!")
                    label = np.roll(label, 10 + int(random.uniform(0, 30)), axis=1)
                else:
                    print("NOT FLIP")


            self.data.set_data(key=img_id, data=(img_id, img, label))

            out = self.data.get_data(key=img_id)

        img_id, img, label = out

       # Random State
        if False:
            self.count = self.count + 1
            torch.manual_seed(self.count)
            random.seed(self.count)

        img2 = img.copy()
        mask = label == 1  

        if False:
            margin = int((256 - self.size[0]) / 2 )
            #margin = 120
            img = img[:, margin:-margin, :]
            label = label[:, margin:-margin, :]

        if False:
            margin = int((256 - self.exp.get_from_config('anisotropy')) / 2 )
            img[:, 0:margin, :] = 0 # = img[:, margin:-margin, :]
            img[:, -margin:256, :] = 0
            label[:, 0:margin, :] = 0 # = img[:, margin:-margin, :]
            label[:, -margin:256, :] = 0
        #img[0:margin, :, :] = 0
        #img[256-margin:256,:,  :] = 0  
        #img[:, 0:margin, :] = 0
        #img[:, 256-margin:256, :] = 0  

        #img = img[margin:-margin, :, :] 
        #label = label[margin:-margin, :, :] 



        #img[:,:,:] = 0
        #label[:,:,:] = 0 

        #img[:, margin:-margin, :] = img2
        #label[:, margin:-margin, :] = label2

        if False:
            img = self.rotate_image(img, 45)
            label = self.rotate_image(label, 45)

        if False:
            img = np.roll(img, self.exp.get_from_config('anisotropy'), axis=1)
            label = np.roll(label, self.exp.get_from_config('anisotropy'), axis=1)
            img[:,0:self.exp.get_from_config('anisotropy'), :] = 0 
            label[:,0:self.exp.get_from_config('anisotropy'), :] = 0 

        if False:
            img2 = cv2.imread("/home/jkalkhof_locale/Downloads/reflection.jpg")
            #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img2, dsize=(256, 256), interpolation=cv2.INTER_CUBIC) 
            #img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2)
            img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img = img2 + img
            img[img > 1]  = 1 
            #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if False:
            img2 = cv2.imread("/home/jkalkhof_locale/Downloads/RedCat.jpg")
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.resize(img2, dsize=(256, 80), interpolation=cv2.INTER_CUBIC) 
            img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2)
            img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img[-80:256, 0:256, :] = img2 

        if False:
            transform = torchio.transforms.RandomBiasField(coefficients = 0.1 * self.exp.get_from_config('anisotropy'), order = 3) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomAnisotropy(axes=1, downsampling=self.exp.get_from_config('anisotropy')) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomNoise(mean = 0, std=0.1)
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            #print("GHOSTING")
            #print(self.exp.get_from_config('anisotropy'))
            transform = torchio.transforms.RandomGhosting(num_ghosts=(self.exp.get_from_config('anisotropy') ,self.exp.get_from_config('anisotropy')), intensity=self.exp.get_from_config('anisotropy')/4, axes=0, restore=0) #num_ghosts=2, intensity=2,
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomAffine(image_interpolation='linear')
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomMotion(image_interpolation='linear')#(image_interpolation='linear')
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomFlip()#(image_interpolation='linear')
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            #torch_rng_state = torch.random.get_rng_state()
            #torch.random.set_rng_state(torch_rng_state)
            transform = torchio.transforms.RandomSpike(num_spikes = 4, intensity=1)#, intensity=0) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        #img = cv2.rectangle(img, (0, 0), (256, 256), (0.8, 0.8, 0.8), 120)

        if False:
            for x in range(5):
                posx = random.randrange(256)
                posy = random.randrange(256)
                while abs(256/2 - posx) < 40 and abs(256/2 - posy) < 40:
                    posx = random.randrange(256)
                    posy = random.randrange(256)              
                img = cv2.circle(img, (posx, posy), 40, (0, 0, 0), -1)

        
        #np.place(img, label > 0, img2[label > 0] )


        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]
        #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = np.clip(img, 0, 1)

        if False:
            plt.imshow(img)
            plt.show()
        return (img_id, img, label)

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

        # REMOVE
        label[label > 0] = 1

        return img, label


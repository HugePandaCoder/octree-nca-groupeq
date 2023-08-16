import cv2
import os
from  src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import numpy as np
import torchvision.transforms as T
import torch
import random


class png_Dataset(Dataset_NiiGz_3D):

    normalize = True

    def __init__(self, crop=False, buffer=False, downscale=4):
        super().__init__()
        self.crop = crop
        self.buffer = buffer
        self.downscale = downscale

    def set_normalize(self, normalize=True):
        self.normalize = normalize

    def load_item(self, path: str) -> np.ndarray:
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.crop:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, dsize=(img.shape[1]//self.downscale,img.shape[0]//self.downscale), interpolation=cv2.INTER_CUBIC)

        img = cv2.convertScaleAbs(img)
        #img = img/256 
        #img = img*2 -1
        #print("MINMAX", torch.max(img), torch.min(img))
        return img

    def __getitem__(self, idx: int) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        if self.buffer:
            img = self.data.get_data(key=self.images_list[idx])

            if not img:
                img_name, p_id, img_id = self.images_list[idx]
                label_name, _, _ = self.labels_list[idx]

                img = self.load_item(os.path.join(self.images_path, img_name))
                label = 0#self.load_item(os.path.join(self.labels_path, img_name))

                self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
                img = self.data.get_data(key=self.images_list[idx])
                img_id, img, label = img
                img = (img_id, img, img)
        else:
            img_name, p_id, img_id = self.images_list[idx]
            label_name, _, _ = self.labels_list[idx]

            img = self.load_item(os.path.join(self.images_path, img_name))
            label = img
            img = (img_id, img, label)


        id, img, label = img

        if self.crop:
            pos_x = random.randint(0, img.shape[0] - self.size[0])
            pos_y = random.randint(0, img.shape[1] - self.size[1])

            img = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
            label = img

        #from matplotlib import pyplot as plt
        #plt.imshow(img[:,:,:])#outputs_fft[0, 0, :, :].real.detach().cpu().numpy())
        #plt.show()

        img = img[...,0:4]

        if self.normalize:
            if False:
                transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                img = torch.from_numpy(img).to(torch.float64)
                img = img.permute((2, 1, 0))
                #print(img.shape)
                img = transform(img)
                img = img.permute((2, 1, 0))
            #img = img * 2 -1
            #img = img
            img = img/128 -1#img/256/2.5 -1  #img/128 -1 #img/256/2.5 -1 #/2.5 -1

        #print(img.shape)

        return (id, img, label)


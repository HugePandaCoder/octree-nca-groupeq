from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import nibabel as nib
import torch
import os
import numpy as np
import cv2
import random

class Dataset_NiiGz_3D_refine(Dataset_NiiGz_3D):
    """Refine results """
    def __getitem__(self, idx, scale=1):
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

        #id, img, label = img#
        #img_array = [img]
        #label_array = [label]

        #for i in range(scale-1):
        #    scaled_size = (int(img_array[-1].shape[0] / 4), int(img_array[-1].shape[1] / 4))#tuple(int(ti/4) for ti in img_array[-1].shape) #(img_array[-1].shape/4)
        #    print(scaled_size)
        #    img_array.insert(0, cv2.resize(img_array[-1], dsize=scaled_size, interpolation=cv2.INTER_CUBIC)) 
        #    label_array.insert(0, cv2.resize(label_array[-1], dsize=scaled_size, interpolation=cv2.INTER_NEAREST))
        #    print(img_array[0].shape)

        return img #(id, img_array, label_array)
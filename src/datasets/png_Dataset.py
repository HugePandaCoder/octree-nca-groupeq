import cv2
import os
from  src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import numpy as np
import torchvision.transforms as T
import torch


class png_Dataset(Dataset_NiiGz_3D):

    def load_item(self, path: str) -> np.ndarray:
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        #img = img/256 
        #img = img*2 -1
        transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = torch.from_numpy(img).to(torch.float64)
        img = img.permute((2, 1, 0))
        #print(img.shape)
        img = transform(img)
        img = img.permute((2, 1, 0))
        #img = img * 2 -1
        #img = img
        img = img/256/2.5 -1
        print("MINMAX", torch.max(img), torch.min(img))
        return img

    def __getitem__(self, idx: int) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        img = self.data.get_data(key=self.images_list[idx])
        if not img:
            img_name, p_id, img_id = self.images_list[idx]
            label_name, _, _ = self.labels_list[idx]

            img = self.load_item(os.path.join(self.images_path, img_name))
            label = self.load_item(os.path.join(self.labels_path, img_name))

            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])

        id, img, label = img

        img = img[...,0:4]

        return (id, img, label)


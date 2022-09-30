from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import nibabel as nib
import torch
import os
import numpy as np
import cv2
import random

class Dataset_NiiGz_3D_advancedPadding(Dataset_NiiGz_3D):
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

    def __getitem__(self, idx, full=False):
        id, img, label = super(Dataset_NiiGz_3D_advancedPadding, self).__getitem__(idx)

        if full == False:
            pad = 32
            img_pad = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            label_pad = np.pad(label, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

            size = self.size

            pos_x = random.randint(0, img_pad.shape[0] - (size[0]+pad*2))
            pos_y = random.randint(0, img_pad.shape[1] - (size[1]+pad*2))

            img = img_pad[pos_x:pos_x+size[0]+pad*2, pos_y:pos_y+size[1]+pad*2, :]
            label = label_pad[pos_x:pos_x+size[0]+pad*2, pos_y:pos_y+size[1]+pad*2, :]

        return (id, img, label)



       
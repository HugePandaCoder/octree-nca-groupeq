from enum import unique
from src.datasets.Dataset_Base import Dataset_Base
from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np

class PascalVOC_Dataset(Dataset_Base):

    def __init__(self):
        super().__init__()
        self.color_label_dic = {}

    def getFilesInPath(self, path):
        r"""Get files in path
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key:file_name, value: file_name}
        """
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            id = f[:-4]
            dic[id] = {}
            dic[id][0] = f
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
            img = cv2.imread(os.path.join(self.images_path, self.images_list[idx]))
            label = cv2.imread(os.path.join(self.labels_path, self.images_list[idx][:-4] + ".png"))
            img, label = self.preprocessing(img, label)


            img_id = "_" + str(img_id)[:-4].replace("_", "") + "_0"
            self.data.set_data(key=img_id, data=(img_id, img, label))

            out = self.data.get_data(key=img_id)
        return out        

    def preprocessing(self, img, label):
        r"""Preprocessing of image
            Args:
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """

        img_scale = img.shape
        min_scale = min(img.shape[0:2])
        min_scale = min_scale / 64
        img_scale = (int(img_scale[0]/min_scale), int(img_scale[1]/min_scale))
        img = cv2.resize(img, dsize=img_scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        label = cv2.resize(label, dsize=img_scale, interpolation=cv2.INTER_NEAREST)
        #label = cv2.normalize(label, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img = img[0:64, 0:64, :]
        label = label[0:64, 0:64, :]

        unique_labels = np.unique(label.reshape(-1, img.shape[2]), axis=0)#np.unique(label, axis=2)
        #print("UNIQUE")
        #print(unique_labels)

        label_mask = np.zeros((label.shape[0], label.shape[1], 24))

        for ul in unique_labels:
            #ul = tuple(map(tuple, ul))
            if str(ul) not in self.color_label_dic:
                self.color_label_dic[str(ul)] = len(self.color_label_dic)

            label_id = self.color_label_dic[str(ul)]
            mask = np.all(label == ul, axis=-1)
            #print(mask.shape)
            label_mask[mask, label_id] = 1
            #print(label_id)
            #print(np.sum(label_mask[:,:, label_id]))


        return img, label_mask
   
        #print(unique_labels)
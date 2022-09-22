from enum import unique
from src.datasets.Dataset_Base import Dataset_Base
from os import listdir
from os.path import isfile, join
import os
import cv2
import numpy as np
import random

from scipy import fftpack
from PIL import Image, ImageDraw

class Cityscapes_Dataset_lowpass(Dataset_Base):

    def __init__(self):
        super().__init__()
        self.color_label_dic = {
            # id, category
            str([0, 0, 0]): [0, 0], # unlabeled, ego vehicle, rectification border, out of roi, static
            str([111, 74, 0]): [1, 0], # dynamic
            str([81, 0, 81]): [2, 0], # ground
            str([128, 64, 128]): [3, 1], # road
            str([244, 35, 232]): [4, 1], # sidewalk
            str([250, 170, 160]): [5, 1], # parking
            str([230, 150, 140]): [6, 1], # rail track
            str([70, 70, 70]): [7, 2], # building
            str([102, 102, 156]): [8, 2], # wall
            str([190, 153, 153]): [9, 2], # fence
            str([180, 165, 180]): [10, 2], # guard rail
            str([150, 100, 100]): [11, 2], # bridge
            str([150, 120, 90]): [12, 2], # tunnel
            str([153, 153, 153]): [13, 3], # pole
            str([250, 170, 30]): [14, 3], # traffic light
            str([220, 220, 0]): [15, 3], # traffic sign
            str([107, 142, 35]): [16, 4], # vegetation
            str([152, 251, 152]): [17, 4], # terrain
            str([70, 130, 180]): [18, 5], # sky
            str([220, 20, 60]): [19, 6], # person
            str([255, 0, 0]): [20, 6], # rider
            str([0, 0, 142]): [21, 7], # car
            str([0, 0, 70]): [22, 7], # truck
            str([0, 60, 100]): [23, 7], # bus
            str([0, 0, 90]): [24, 7], # caravan
            str([0, 0, 110]): [25, 7], # trailer
            str([0, 80, 100]): [26, 7], # train
            str([0, 0, 230]): [27, 7], # motorcycle
            str([119, 11, 32]): [28, 7], # bicycle
            str([0, 0, 142]): [29, 7], # license plate
        }

    def lowpass_filter(self, image):
        #image = image.numpy()
        fft1 = fftpack.fftshift(fftpack.fft2(image))

        #print("FFT!")
        #print(fft1.shape)

        x,y = image.shape[0], image.shape[1]

        #scale_x, scale_y = image.shape[0]/100, image.shape[1]/100

        min_scale = min(image.shape)/100

        #size of circle
        e_x,e_y=300, 300#200*scale_x,200*scale_y
        #create a box 
        bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))
        low_pass=Image.new("L",(image.shape[1],image.shape[0]),color=0)

        draw1=ImageDraw.Draw(low_pass)
        
        #bbox= 
        draw1.rectangle(bbox, fill=1)
        
        #draw1.ellipse(bbox, fill=1)

        low_pass_np=1 - np.array(low_pass) #1-

        #multiply both the images
        filtered=np.multiply(fft1,low_pass_np)

        #inverse fft
        ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
        #ifft2 = np.maximum(0, np.minimum(ifft2, 255))

        #ifft2 = torch.from_numpy(ifft2)

        return ifft2

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = cv2.imread(os.path.join(self.labels_path, self.images_list[idx][:-4] + ".png"))
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
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

        #img_scale = img.shape
        #min_scale = min(img.shape[0:2])
        #min_scale = min_scale / 64
        #img_scale = (int(img_scale[0]/min_scale), int(img_scale[1]/min_scale))
        #img = cv2.resize(img, dsize=img_scale, interpolation=cv2.INTER_CUBIC)
        
        for z in range(img.shape[2]):
            #print(img.shape)
            img[:, :, z] = self.lowpass_filter(img[:, :, z].copy())

        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #label = cv2.resize(label, dsize=img_scale, interpolation=cv2.INTER_NEAREST)

        pos_x = random.randint(0, img.shape[0] - self.size[0])
        pos_y = random.randint(0, img.shape[1] - self.size[1])

        img = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
        label = label[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]

        unique_labels = np.unique(label.reshape(-1, img.shape[2]), axis=0)#np.unique(label, axis=2)
        #print("UNIQUE")
        #print(unique_labels)

        label_mask = np.zeros((label.shape[0], label.shape[1], 24))

        for ul in unique_labels:
            #ul = tuple(map(tuple, ul))
            #if str(ul) not in self.color_label_dic:
            #    self.color_label_dic[str(ul)] = len(self.color_label_dic)
            ul = ul.tolist()
            label_id = self.color_label_dic[str(ul)][1]
            mask = np.all(label == ul, axis=-1)
            #print(mask.shape)
            label_mask[mask, label_id] = 1
            #print(label_id)
            #print(np.sum(label_mask[:,:, label_id]))


        return img, label_mask
   
        #print(unique_labels)
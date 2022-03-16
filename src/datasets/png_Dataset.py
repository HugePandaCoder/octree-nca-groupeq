from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import nibabel as nib
import sys
import numpy as np
import cv2

class png_Dataset(Dataset):

    def __init__(self, root_dir, size):
        self.root_dir = root_dir
        self.images_list = listdir(join(root_dir, "images"))
        self.length = [f for f in listdir(join(root_dir, "images")) if isfile(join(join(root_dir, "images"), f))]
        self.size = size


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        #print(join(self.root_dir, "images", self.images_list[idx]))
        #print(join(self.root_dir, "images", self.images_list[idx]))
        img = cv2.imread(join(self.root_dir, "images", self.images_list[idx]))
        #print(img)
        img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) #[..., np.newaxis] 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)/256
        img[:,:,3] = 1
        #print(img.shape)

        label_path = join(self.root_dir, "trimaps", self.images_list[idx])[:-4] + ".png"
        label = cv2.imread(label_path)
        if(label != None):
            label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST) #[..., np.newaxis] 
            label = cv2.cvtColor(label, cv2.COLOR_RGB2RGBA)
            label[:,:, 3] = np.clip(np.round(label[:,:, 0]),0,1)

            label[:,:, 0] = label[:,:, 0] == 2
            label[:,:, 1] = label[:,:, 1] == 1
            label[:,:, 2] = label[:,:, 2] == 3


        #print(img.shape)
        #print(label.shape)

        return img, label


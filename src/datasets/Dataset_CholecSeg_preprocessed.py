

from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Dataset_Base import Dataset_Base
import os
import numpy as np
import cv2
import torch, math

class Dataset_CholecSeg_preprocessed(Dataset_Base):
    def __init__(self):
        super().__init__()
        self.slice = None
        self.delivers_channel_axis = True
        self.is_rgb = True


    
    def getFilesInPath(self, path: str):
        r"""Get files in path
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key: patient_name, value: {key: index, value: file_name}}
        """
        dirs = os.listdir(path)
        dic = {}
        for inner_dir in dirs:
            if not os.path.isdir(os.path.join(path, inner_dir)):
                continue
            id = inner_dir
            dic[id] = {}
            for i, f in enumerate(os.listdir(os.path.join(path, inner_dir))):
                dic[id][i] = f
        return dic
    
    def __getitem__(self, idx: str):
        # images have a resolution of 854x480 with 80 frames
        patient_name = self.images_list[idx][:len("videoXX")]
        path = os.path.join(self.images_path, patient_name, self.images_list[idx])

        imgs = np.load(os.path.join(path, "video.npy"))#CHWD
        lbls = np.load(os.path.join(path, "segmentation.npy"))#HWDC

        def reshape_batch(instack, is_label:bool = False) -> np.ndarray:
            #https://stackoverflow.com/questions/65154879/using-opencv-resize-multiple-the-same-size-of-images-at-once-in-python
            N,H,W,C = instack.shape
            instack = instack.transpose((1,2,3,0)).reshape((H,W,C*N))

            outstacks = []
            for i in range(math.ceil(instack.shape[-1] / 500)):
                if is_label:
                    outstack = cv2.resize(instack[..., i*500:(i+1)*500], (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    outstack = cv2.resize(instack[..., i*500:(i+1)*500], (self.size[1], self.size[0]))
                outstacks.append(outstack)

            outstack = np.concatenate(outstacks, axis=-1)
            return outstack.reshape((self.size[0], self.size[1], C, N)).transpose((3,0,1,2))


        imgs = reshape_batch(imgs.transpose(3,1,2,0))
        lbls = reshape_batch(lbls.transpose(2,0,1,3), is_label=False)

        imgs = imgs.transpose(3, 1, 2, 0)#DHWC -> CHWD
        lbls = lbls.transpose(1, 2, 0, 3)#DHWC -> HWDC

        data_dict = {}
        data_dict['id'] = self.images_list[idx]
        data_dict['image'] = imgs
        data_dict['label'] = lbls
        return data_dict
    
    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        return super().setPaths(images_path, images_list, labels_path, labels_list)
    
    def set_size(self, size: tuple) -> None:
        super().set_size(size)
        assert self.size[2] == 80, "The temporal dimension must be 80"
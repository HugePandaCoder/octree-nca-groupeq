

from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Dataset_Base import Dataset_Base
import os
import numpy as np
import cv2
import torch, math

class Dataset_CholecSeg_preprocessed(Dataset_Base):
    def __init__(self, use_max_sequence_length_in_eval: bool, patch_size: tuple=None):
        super().__init__()
        self.slice = None
        self.delivers_channel_axis = True
        self.is_rgb = True
        self.use_max_sequence_length_in_eval = use_max_sequence_length_in_eval

        self.patch_size = patch_size

    
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
    


    def load_item_internal(self, path):
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
        data_dict['image'] = imgs
        data_dict['label'] = lbls
        return data_dict

    def setState(self, state: str) -> None:
        super().setState(state)

    def __getitem__(self, idx: str):
        # images have a resolution of 854x480 with 80 frames

        if self.state == "train" or not self.use_max_sequence_length_in_eval:
            patient_name = self.images_list[idx][:len("videoXX")]
            path = os.path.join(self.images_path, patient_name, self.images_list[idx])

            data_dict = self.load_item_internal(path)


            data_dict['id'] = self.images_list[idx]
        else:
            assert self.state in ["val", "test"]
            patient_name = self.images_list[idx][:len("videoXX")]
            frame_end = int(self.images_list[idx][len("videoXX") + 1:])
            all_folders = []

            iterator = 0
            while True:
                path_at_question = os.path.join(self.images_path, patient_name, f"{patient_name}_{str(frame_end + iterator).zfill(5)}")
                if os.path.exists(path_at_question):
                    all_folders.append(path_at_question)
                    iterator += 80
                else:
                    break
            iterator = 80
            while True:
                path_at_question = os.path.join(self.images_path, patient_name, f"{patient_name}_{str(frame_end - iterator).zfill(5)}")
                if os.path.exists(path_at_question):
                    all_folders.append(path_at_question)
                    iterator += 80
                else:
                    break

            all_folders.sort()
            all_segmentations = []
            all_videos = []
            for folder in all_folders:
                temp_dict = self.load_item_internal(folder)
                all_videos.append(temp_dict['image'])
                all_segmentations.append(temp_dict['label'])

            data_dict = {}
            data_dict['id'] = self.images_list[idx]
            data_dict['image'] = np.concatenate(all_videos, axis=3)
            data_dict['label'] = np.concatenate(all_segmentations, axis=2)


        
        if self.patch_size is not None and self.state == "train":
            img = data_dict['image']
            lbl = data_dict['label']
            if img.shape[1] == self.patch_size[0]:
                x = 0
            else:
                x = np.random.randint(0, img.shape[1] - self.patch_size[0])
            
            if img.shape[2] == self.patch_size[1]:
                y = 0
            else:
                y = np.random.randint(0, img.shape[2] - self.patch_size[1])

            if img.shape[3] == self.patch_size[2]:
                z = 0
            else:
                z = np.random.randint(0, img.shape[3] - self.patch_size[2])
            
            img = img[:, x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]]
            lbl = lbl[x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]] 
            data_dict['image'] = img
            data_dict['label'] = lbl


        return data_dict
    
    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        return super().setPaths(images_path, images_list, labels_path, labels_list)
    
    def set_size(self, size: tuple) -> None:
        super().set_size(size)
        assert self.size[2] == 80, "The temporal dimension must be 80"
from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import torch
import os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import random
import torchio

class Dataset_NiiGz_3D(Dataset_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """

    def getDataShapes():
        return

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

    def getSlicesOnAxis(self, path, axis):
        return self.load_item(path).shape[axis]

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            Args:
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def __getitem__(self, idx):
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

            if len(img.shape) == 4:
                img = img[...,0] 

            #size = (256, 256) 
            


            #resize_factor = (size[0] / img.shape[0], size[1] / img.shape[1], 1)
            #print(resize_factor) 

            #img = scipy.ndimage.interpolation.zoom(img, resize_factor)
            #label = scipy.ndimage.interpolation.zoom(label, resize_factor, order=0)
            
            #print(img.shape)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #img =img[20:-20, :, :] 
            #plt.imshow(img)
            #plt.show()
            
            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])

        id, img, label = img

        size = self.size #(256, 256)#
        size = (256, 256)

        img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC) 
        label = cv2.resize(label, dsize=size, interpolation=cv2.INTER_NEAREST) 

        margin = int((256 - self.size[0]) / 2 )

        #img[0:margin, :, :] = 0
        #img[256-margin:256,:,  :] = 0  
        #img[:, 0:margin, :] = 0
        #img[:, 256-margin:256, :] = 0  

        #img = img[margin:-margin, :, :] 
        #label = label[margin:-margin, :, :] 

        img = img[:, margin:-margin, :]
        label = label[:, margin:-margin, :]

        #img[:,:,:] = 0
        #label[:,:,:] = 0 

        #img[:, margin:-margin, :] = img2
        #label[:, margin:-margin, :] = label2

        
        #img = self.rotate_image(img, 4 5)

        #img = np.roll(img, self.exp.get_from_config('anisotropy'), axis=1)
        #label = np.roll(label, self.exp.get_from_config('anisotropy'), axis=1)

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
            transform = torchio.transforms.RandomAnisotropy(axes=1, downsampling=self.exp.get_from_config('anisotropy')) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            torch.manual_seed(42)
            transform = torchio.transforms.RandomSpike(num_spikes=1, intensity=16) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)
        
        
        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]

        #img = cv2.rectangle(img, (0, 0), (256, 256), (0.8, 0.8, 0.8), 120)

        #for x in range(10):
        #    posx = random.randrange(256)
        #    posy = random.randrange(256)
        #    img = cv2.circle(img, (posx, posy), 10, (1, 1, 1), -1)

        #print(np.max(img))

        plt.imshow(img)
        plt.show()

        # REMOVE
        label[label > 0] = 1

        return (id, img, label)

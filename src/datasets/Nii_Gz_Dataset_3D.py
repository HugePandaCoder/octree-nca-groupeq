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
                dic[id][0] = (f, f, 0)           
        return dic

    def getSlicesOnAxis(self, path, axis):
        return self.load_item(path).shape[axis]

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            Args:
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()

    def rotate_image(self, image, angle, label = False):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        if label:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
        else:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def preprocessing3d(self, img, isLabel=False):
        if len(img.shape) == 4:
            img = img[..., 0]
        padded = np.zeros((400,400,64))# ((64, 64, 52)) #(400,400,64))
        img_shape = img.shape
        padded[0:img_shape[0], 0:img_shape[1], 0:img_shape[2]] = img
        #print(padded.shape)

        # For now single mask
        if isLabel == True:
            padded[padded != 0] = 1

        return padded

    def __getitem__(self, idx):
        r"""Standard get item function
            Args:
                idx (int): Id of item to loa
            Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        znormalisation = torchio.ZNormalization()
        #histogram_stanard = torchio.HistogramStandardization()

        img = self.data.get_data(key=self.images_list[idx])
        if not img:
            #print(self.images_list[idx])
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
                # Remove 4th dimension
                if len(img.shape) == 4:
                    img = img[...,0] 
                img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
            else:
                img, label = self.preprocessing3d(img), self.preprocessing3d(label, isLabel=True)
            img_id = "_" + str(p_id) + "_" + str(img_id)



            #size = (256, 256) 
            


            #resize_factor = (size[0] / img.shape[0], size[1] / img.shape[1], 1)
            #print(resize_factor) 

            #img = scipy.ndimage.interpolation.zoom(img, resize_factor)
            #label = scipy.ndimage.interpolation.zoom(label, resize_factor, order=0)
            
            #print(img.shape)
            img = np.expand_dims(img, axis=0)
            img = rescale(img) #cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = np.squeeze(img)
            #img =img[20:-20, :, :] 
            #plt.imshow(img)
            #plt.show()
            
            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])

        id, img, label = img
        size = self.size #(256, 256)#

        if len(size) > 2:
            size = size[0:2] 
        #size = (256, 256)

        img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC) 
        label = cv2.resize(label, dsize=size, interpolation=cv2.INTER_NEAREST) 

        # Random State
        self.count = self.count + 1
        torch.manual_seed(self.count)
        random.seed(self.count)

        img2 = img.copy()
        mask = label == 1  

        if False:
            margin = int((256 - self.size[0]) / 2 )
            img = img[:, margin:-margin, :]
            label = label[:, margin:-margin, :]
        #img[0:margin, :, :] = 0
        #img[256-margin:256,:,  :] = 0  
        #img[:, 0:margin, :] = 0
        #img[:, 256-margin:256, :] = 0  

        #img = img[margin:-margin, :, :] 
        #label = label[margin:-margin, :, :] 



        #img[:,:,:] = 0
        #label[:,:,:] = 0 

        #img[:, margin:-margin, :] = img2
        #label[:, margin:-margin, :] = label2

        if False:
            img = self.rotate_image(img, 45)
            label = self.rotate_image(label, 45)

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
            transform = torchio.transforms.RandomBiasField() #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomAnisotropy(axes=1, downsampling=8) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomNoise(mean = 0.2, std=0.05)
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomGhosting()
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomAffine(image_interpolation='linear')
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomMotion()#(image_interpolation='linear')
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            transform = torchio.transforms.RandomFlip()#(image_interpolation='linear')
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        if False:
            #torch_rng_state = torch.random.get_rng_state()
            #torch.random.set_rng_state(torch_rng_state)
            transform = torchio.transforms.RandomSpike(intensity=1)#, intensity=0) #, **kwargs()#(num_ghosts=10, axes=(0, 1, 2), intensity=(3, 5))
            img = np.expand_dims(img, axis=0)
            img = transform(img)
            img = np.squeeze(img)

        #img = cv2.rectangle(img, (0, 0), (256, 256), (0.8, 0.8, 0.8), 120)

        if False:
            for x in range(5):
                posx = random.randrange(256)
                posy = random.randrange(256)
                while abs(256/2 - posx) < 40 and abs(256/2 - posy) < 40:
                    posx = random.randrange(256)
                    posy = random.randrange(256)              
                img = cv2.circle(img, (posx, posy), 40, (0, 0, 0), -1)

        
        #np.place(img, label > 0, img2[label > 0] )


        img[..., 1] = img[..., 0]
        img[..., 2] = img[..., 0]

        #transform = torchio.transforms.RandomSpike(intensity=1)#torchio.transforms.RandomElasticDeformation()
        if self.state == "train" and random.randrange(0, 2) == 1 and False:
            rand = random.randrange(0, 10)

            if rand == 0:
                #print("TRAIN")
                transform = torchio.transforms.RandomSpike(intensity=1)
                img = np.expand_dims(img, axis=0)
                img = transform(img)
                img = np.squeeze(img)
            if rand == 1:
                #print("TRAIN")
                transform = transform = torchio.transforms.RandomFlip()
                img = np.expand_dims(img, axis=0)
                img = transform(img)
                img = np.squeeze(img) #
            if rand == 2:
                #print("TRAIN")
                transform = torchio.transforms.RandomNoise(mean = 0.2, std=0.05)
                img = np.expand_dims(img, axis=0)
                img = transform(img)
                img = np.squeeze(img) #            transform = torchio.transforms.RandomBiasField() 
            if rand == 3:
                #print("TRAIN")
                transform = torchio.transforms.RandomBiasField() 
                img = np.expand_dims(img, axis=0)
                img = transform(img)
                img = np.squeeze(img)
            if rand == 4:
                rnd_seed = random.randint(0, 1000000)
                transform = torchio.transforms.RandomAffine()
                random.seed(rnd_seed)
                img = np.expand_dims(img, axis=0)
                img = transform(img)
                img = np.squeeze(img)
                transform = torchio.transforms.RandomAffine(image_interpolation='nearest', label_interpolation='nearest')
                random.seed(rnd_seed)
                label = np.expand_dims(label, axis=0)
                label = transform(label)
                label = np.squeeze(label)
            if rand >= 5:
                rnd_seed = random.randint(0, 1000000)
                transform = torchio.transforms.RandomElasticDeformation()
                random.seed(rnd_seed)
                img = np.expand_dims(img, axis=0)
                img = transform(img)
                img = np.squeeze(img)
                transform = torchio.transforms.RandomElasticDeformation(image_interpolation='nearest', label_interpolation='nearest')
                random.seed(rnd_seed)
                label = np.expand_dims(label, axis=0)
                label = transform(label)
                label = np.squeeze(label)

        img = np.expand_dims(img, axis=0)
        img = znormalisation(img)
        #img = histogram_stanard(img)
        img = rescale(img) #cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = np.squeeze(img)
        #img = np.clip(img, 0, 1)

        if False:
            plt.imshow(img[:,:,26])
            plt.show()
        # REMOVE
        label[label > 0] = 1

        return (id, img, label)

from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio
import src.numpyio.RescaleIntensity as NioRescaleIntensity
import src.numpyio.ZNormalization as NioZNormalization

def my_print(*args, **kwargs):
    pass
    #print(*args, **kwargs)

class Dataset_NiiGz_3D(Dataset_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """

    def getDataShapes(self) -> None:
        return

    def getFilesInPath(self, path: str) -> dict:
        r"""Get files in path ordered by id and slice
            #Args
                path (string): The path which should be worked through
            #Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = os.listdir(os.path.join(path))
        dic = {}
        for id_f, f in enumerate(dir_files):
            id = f
            # 2D 
            if self.slice is not None:
                for slice in range(self.getSlicesOnAxis(os.path.join(path, f), self.slice)):
                    if id not in dic:
                        dic[id] = {}
                    dic[id][slice] = (f, id_f, slice)
            # 3D
            else:
                if id not in dic:
                    dic[id] = {}
                dic[id][0] = (f, f, 0)
        return dic

    def getSlicesOnAxis(self, path: str, axis: int) -> nib.nifti1:
        return self.load_item(path).shape[axis]

    def load_item(self, path: str) -> nib.nifti1:
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()

    def rotate_image(self, image: np.ndarray, angle: float, label: bool = False) -> np.ndarray:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        if label:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
        else:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def preprocessing3d(self, img: np.ndarray, isLabel: bool = False) -> np.ndarray:
        r"""Preprocess data to fit the required shape
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if not isLabel:
            # TODO: Currently only single volume, no multi phase
            if len(img.shape) == 4:
                img = img[..., 0]
            padded = np.random.rand(*self.size) * 0.01
        else:
            padded = np.zeros(self.size)
        img_shape = img.shape
        padded[0:img_shape[0], 0:img_shape[1], 0:img_shape[2]] = img

        return padded

    def rescale3d(self, img: np.ndarray, isLabel: bool = False) -> np.ndarray:
        r"""Rescale input image to fit training size
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if len(self.size) == 3:
            size = (self.size[1], self.size[0])
            size2 = (self.size[2], self.size[0])
        else:
            size = (self.size[0], self.size[1])

        img_resized = np.zeros((self.size[0], self.size[1], img.shape[2])) 
        for x in range(img.shape[2]):
            if not isLabel:
                img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=size, interpolation=cv2.INTER_CUBIC) 
            else:
                img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=size, interpolation=cv2.INTER_NEAREST) 

        if len(self.size) == 3 and True:
            img = img_resized
            img_resized = np.zeros((self.size[0], self.size[1], self.size[2]))
            for x in range(img.shape[1]):
                if not isLabel:
                    img_resized[:, x, :] = cv2.resize(img[:, x, :], dsize=size2, interpolation=cv2.INTER_CUBIC) 
                else:
                    img_resized[:, x, :] = cv2.resize(img[:, x, :], dsize=size2, interpolation=cv2.INTER_NEAREST) 

        return img_resized

    def patchify(self, img: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Take a patch of the input. This should be used instead of rescaling if global information is not required.
            #Args
                img (numpy): Image data
                label (numpy): Label data
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        size = self.exp.config['experiment.dataset.patchify.patch_size']

        enforce_mask = (random.uniform(0, 1) < self.exp.get_from_config('experiment.dataset.patchify.foreground_oversampling_probability'))
        while True:
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])

            if enforce_mask:
                if 1 in np.unique(label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]):
                    break
            else: 
                break
        
        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]

        return img, label
    
    def badLabels(self, label: np.ndarray, shifts: tuple = None) -> np.ndarray:
        r"""Create artifically badly labbelled data
            #Args
                label (numpy): Label data
            #Returns:
                label (numpy): Label data
        """
        if shifts is None:
            shift_x = random.randint(10, 30)
            shift_y = random.randint(10, 30)
            shift_z = random.randint(10, 30)
            if random.randint(0, 2) == 1:
                shift_x = shift_x * -1
            if random.randint(0, 2) == 1:
                shift_y = shift_y * -1
            if random.randint(0, 2) == 1:
                shift_z = shift_z * -1
        else:
            shift_x, shift_y, shift_z = shifts


        print(shift_x, shift_y, shift_z)


        label = np.roll(label, shift_x, axis=0)
        label = np.roll(label, shift_y, axis=1)
        label = np.roll(label, shift_z, axis=2)

        return label


    def randomReplaceByNoise(self, img: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Replace parts of the image by noise
            #Args
                img (numpy): Image data
                label (numpy): Label data
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        axis = random.randint(0, 2)
        side = random.randint(0, 2)
        slides = random.randint(0, int(img.shape[axis]/3)) 

        if side == 0 or side == 2:
            if axis == 0:
                img[0:slides, :, :] = np.random.rand(slides, img.shape[1], img.shape[2]) * 0.01
            if axis == 1:
                img[:, 0:slides, :] = np.random.rand(img.shape[0], slides, img.shape[2]) * 0.01
            if axis == 2:
                img[:, :, 0:slides] = np.random.rand(img.shape[0], img.shape[1], slides) * 0.01
        if side == 1 or side == 2:
            if axis == 0:
                img[-slides-1:-1, :, :] = np.random.rand(slides, img.shape[1], img.shape[2]) * 0.01
            if axis == 1:
                img[:, -slides-1:-1, :] = np.random.rand(img.shape[0], slides, img.shape[2]) * 0.01
            if axis == 2:
                img[:, :, -slides-1:-1] = np.random.rand(img.shape[0], img.shape[1], slides) * 0.01

        if side == 0 or side == 2:
            if axis == 0:
                label[0:slides, :, :] = 0
            if axis == 1:
                label[:, 0:slides, :] = 0
            if axis == 2:
                label[:, :, 0:slides] = 0
        if side == 1 or side == 2:
            if axis == 0:
                label[-slides-1:-1, :, :] = 0
            if axis == 1:
                label[:, -slides-1:-1, :] = 0
            if axis == 2:
                label[:, :, -slides-1:-1] = 0

        return img, label
        

    def getPublicIdByIndex(self, idx: int):
        img_name, p_id, img_id = self.images_list[idx]
        img_id = "_" + str(p_id) + "_" + str(img_id)
        return img_id 


    def __getitem__(self, idx: str) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        my_print("Dataset_NiiGz_3D line 222, idx: ", idx)
        #rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        #znormalisation = torchio.ZNormalization()
        rescale = NioRescaleIntensity.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        znormalisation = NioZNormalization.ZNormalization()


        img = self.data.get_data(key=self.images_list[idx])
        my_print("Dataset_NiiGz_3D line 227")
        if not img:
            my_print("Dataset_NiiGz_3D line 229")
            img_name, p_id, img_id = self.images_list[idx]

            label_name, _, _ = self.labels_list[idx]

            img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, img_name))

            my_print("Dataset_NiiGz_3D line 235")
            # Augmentations
            if self.augment is not None:
                assert False, "Augmentations not implemented"
                print("AUGMENTATION " + self.augment)

                img_tio = torchio.ScalarImage(tensor=img)
                if self.augment == "spike":
                    spike_augmentation = torchio.transforms.RandomSpike(num_spikes=1, intensity=(0.3, 0.3))
                    img_tio = spike_augmentation(img_tio)
                else:
                    raise ValueError("Augmentation not implemented")

                img = img_tio.tensor.numpy()


            my_print("Dataset_NiiGz_3D line 255, slice:", self.slice)
            # 2D
            if self.slice is not None:
                if len(img.shape) == 4:
                    img = img[..., 0]
                if self.exp.get_from_config('experiment.dataset.rescale') is not None and self.exp.get_from_config('experiment.dataset.rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.slice == 0:
                    img, label = img[img_id, :, :], label[img_id, :, :]
                elif self.slice == 1:
                    img, label = img[:, img_id, :], label[:, img_id, :]
                elif self.slice == 2:
                    img, label = img[:, :, img_id], label[:, :, img_id]
                # Remove 4th dimension if multiphase
                if len(img.shape) == 4:
                    img = img[...,0] 
                img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
            # 3D
            else:
                my_print("Dataset_NiiGz_3D line 270")
                if len(img.shape) == 4:
                    img = img[..., 0]
                img = np.expand_dims(img, axis=0)
                my_print("Dataset_NiiGz_3D line 274, mean img:",np.mean(img))
                img = rescale(img) 
                my_print("Dataset_NiiGz_3D line 276")
                img = np.squeeze(img)
                my_print("Dataset_NiiGz_3D line 278")
                if self.exp.get_from_config('experiment.dataset.rescale') is not None and self.exp.get_from_config('experiment.dataset.rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.exp.get_from_config('experiment.dataset.keep_original_scale') is not None and self.exp.get_from_config('experiment.dataset.keep_original_scale'):
                    img, label = self.preprocessing3d(img), self.preprocessing3d(label, isLabel=True)  
                # Add dim to label
                my_print("Dataset_NiiGz_3D line 282")
                if len(label.shape) == 3:
                    label = np.expand_dims(label, axis=-1)
            slice_id = str(img_id)
            img_id = "_" + str(p_id) + "_" + str(img_id)

            my_print("Dataset_NiiGz_3D line 284")
            if self.store:
                self.data.set_data(key=self.images_list[idx], data=(img_id, img, label, str(p_id), slice_id))
                img = self.data.get_data(key=self.images_list[idx])
            else:
                img = (img_id, img, label, str(p_id), slice_id)
           
        my_print("Dataset_NiiGz_3D line 291")
        

        id, img, label, p_id, slice_id = img

        size = self.size 
        
        # Create patches from full resolution
        if self.exp.get_from_config('experiment.dataset.patchify') is not None and self.exp.get_from_config('experiment.dataset.patchify') is True and self.state == "train": 
            img, label = self.patchify(img, label) 

        if len(size) > 2:
            size = size[0:2] 

        # Normalize image
        img = np.expand_dims(img, axis=0)
        if np.sum(img) > 0:
            img = znormalisation(img)
        img = rescale(img) 
        img = img[0]


        if self.exp.get_from_config('model.output_channels') > 1:
            label = np.eye(self.exp.get_from_config('model.output_channels')+1)[label.astype(np.int32)].squeeze()
            label = label[..., 1:label.shape[-1]]
        else:
            # Merge labels -> For now single label
            label[label > 0] = 1


        # Number of defined channels
        if len(self.size) == 2:
            img = img[..., :self.exp.get_from_config('model.input_channels')]
            label = label[..., :self.exp.get_from_config('model.output_channels')]

        data_dict = {}
        data_dict['id'] = id
        data_dict['patient_id'] = p_id
        data_dict['recording_id'] = p_id#each patient has only one recording
        if self.slice is not None:
            data_dict['slice_id'] = slice_id
        data_dict['image'] = img
        data_dict['label'] = label

        #print(label.shape)


        return data_dict

from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import os

class Dataset_NiiGz_3D(Dataset_3D):

    def getDataShapes():
        return

    r"""Get files in path ordered by id and slice
        Args:
            path (string): The path which should be worked through
        Returns:
            dic (dictionary): {key:patientID, {key:sliceID, img_slice}
    """
    def getFilesInPath(self, path):
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
        return nib.load(path).get_fdata()

    r"""Get number of items in dataset"""
    def __len__(self):
        return self.length

    r"""Standard get item function
        Args:
            idx (int): Id of item to loa
        Returns:
            img (numpy): Image data
            label (numpy): Label data
    """
    def __getitem__(self, idx):
        #
        # print(self.images_list)
        img_name, p_id, img_id = self.images_list[idx]
        label_name, _, _ = self.labels_list[idx]
        img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, label_name))
        if self.slice is not None:
            if self.slice == 0:
                img, label = img[img_id, :, :], label[img_id, :, :]
            elif self.slice == 1:
                img, label = img[:, img_id, :], label[:, img_id, :]
            elif self.slice == 2:
                img, label = img[:, :, img_id], label[:, :, img_id]
            img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
        img_id = "_" + str(p_id) + "_" + str(img_id)
        return img_id, img, label 
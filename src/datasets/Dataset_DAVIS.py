

from src.datasets.Dataset_3D import Dataset_3D
from src.datasets.Dataset_Base import Dataset_Base


class Dataset_DAVIS(Dataset_Base):
    
    def getFilesInPath(self, path: str):
        return super().getFilesInPath(path)
    
    def __getitem__(self, idx: str):
        return super().__getitem__(idx)
    
    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str) -> None:
        print("Setting paths", images_path, images_list, labels_path, labels_list)
        exit()
        return super().setPaths(images_path, images_list, labels_path, labels_list)
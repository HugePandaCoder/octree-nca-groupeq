
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np

from src.datasets import Dataset_Base
from src.datasets.BatchgeneratorsDataLoader import my_default_collate
from src.utils.DataAugmentations import get_transform_arr
import math
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose

def my_print(*args, **kwargs):
    pass
    #print(*args, **kwargs)


class DatasetPerEpochGenerator(SlimDataLoaderBase):
    #https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/multithreaded_with_batches.ipynb
    def __init__(self, data, num_threads_in_mt=12, batch_size=4):
        # This initializes self._data, self.batch_size and self.number_of_threads_in_multithreaded
        super(DatasetPerEpochGenerator, self).__init__(data, batch_size, num_threads_in_mt)

        self.num_restarted = 0
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.data_indices = np.arange(len(self._data))
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_indices)
        self.was_initialized = True
        self.num_restarted = self.num_restarted + 1
        self.current_position = self.thread_id*self.batch_size

    def generate_train_batch(self):
        my_print("intiliazed: ", self.was_initialized)
        if not self.was_initialized:
            self.reset()
        my_print("DatasetPerEpochGenerator line 35")
        idx = self.current_position
        if idx < len(self._data):
            my_print("DatasetPerEpochGenerator line 38")
            self.current_position = idx + self.batch_size*self.number_of_threads_in_multithreaded
            indices = self.data_indices[idx: min(len(self._data),idx+self.batch_size)]
            my_print(f"DatasetPerEpochGenerator line 41, indices: {indices}")
            batch = [self._data[i] for i in indices]
            my_print([type(b) for b in batch])
            batch = my_default_collate(batch)
            my_print("DatasetPerEpochGenerator line 41")
            return batch
        else:
            my_print("DatasetPerEpochGenerator line 44")
            self.was_initialized=False
            raise StopIteration

class StepsPerEpochGenerator(SlimDataLoaderBase):
    def __init__(self, data, num_steps_per_epoch:int, num_threads_in_mt=12, batch_size=4):
        # This initializes self._data, self.batch_size and self.number_of_threads_in_multithreaded
        super(StepsPerEpochGenerator, self).__init__(data, batch_size, num_threads_in_mt)
        self.num_steps_per_epoch = num_steps_per_epoch
        self.counter = 0
    def generate_train_batch(self):
        if self.counter > self.num_steps_per_epoch:
            raise StopIteration
        self.counter += 1
        indices = np.random.choice(np.arange(len(self._data)), self.batch_size)
        batch = [self._data[i] for i in indices]
        batch = my_default_collate(batch)
        return batch


class MultiThreadedAugmenterWithLength(MultiThreadedAugmenter):
    def __len__(self):
        if hasattr(self, 'length'):
            return self.length
        else:
            return -1
        
def get_batchgenerators_dataset(dataset_class, num_processes: int, num_steps_per_epoch: int, batch_size: int):
    class BatchgeneratorsDataset(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if num_steps_per_epoch is None:
                generator = DatasetPerEpochGenerator(self, num_processes, batch_size)
            else:
                generator = StepsPerEpochGenerator(self, num_steps_per_epoch, num_processes, batch_size)
            transforms = get_transform_arr()
            transforms.append(NumpyToTensor(keys=['image', 'label']))
            self.dataloader = MultiThreadedAugmenterWithLength(generator, Compose(transforms), num_processes=num_processes)
        
        def setDataloaderLength(self):
            if num_steps_per_epoch is None:
                self.dataloader.length = math.ceil(len(self) / batch_size)
            else:
                self.dataloader.length = num_steps_per_epoch


        def setState(self, state: str) -> None:
            if hasattr(self, 'state'):
                state_changed = self.state != state
            else:
                state_changed = True
            super().setState(state)
            if state_changed:
                self.setDataloaderLength()
                self.dataloader.restart()

        def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str):
            if hasattr(self, 'images_path'):
                paths_changed = self.images_path != images_path or self.images_list != images_list or self.labels_path != labels_path or self.labels_list != labels_list
            else:
                paths_changed = True
            super().setPaths(images_path, images_list, labels_path, labels_list)
            if paths_changed:
                self.setDataloaderLength()
                self.dataloader.restart()

        def __iter__(self):
            return self.dataloader


    return BatchgeneratorsDataset


"""
class BatchgeneratorsDatasetWrapperDataset:
    def __init__(self, dataset_class, num_processes: int, num_steps_per_epoch) -> None:
        self.dataset_class = dataset_class
        self.num_processes = num_processes
        self.num_steps_per_epoch = num_steps_per_epoch

        self.currently_active_key = None

        self.datasets = dict()
        self.dataloaders = dict()


    def __len__(self) -> int:
        return len(self.datasets[self.currently_active_key])

    def __getitem__(self, index: int) -> tuple:
        return next(self.datasets[self.currently_active_key][index])

    def change_key(self, key: tuple) -> None:
        self.currently_active_key = key
        if key not in self.datasets:
            self.datasets[key] = self.dataset_class()
            self.datasets[key].setPaths(*(key[:-1]))
            self.datasets[key].setState(key[-1])
            #TODO generator = 
            self.dataloaders[key] = MultiThreadedAugmenter(generator, get_transform_arr(), num_processes=self.num_processes)
            self.dataloaders[key].

    def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str):
        key = [images_path, images_list, labels_path, labels_list, 'train']
        if key not in self.datasets:
            self.datasets[key] = self.dataset_class()
            self.datasets[key].setPaths(images_path, images_list, labels_path, labels_list)
            self.dataset_states[key] = 'train'
            self.dataloaders[key] = MultiThreadedAugmenter(self.datasets[key], get_transform_arr(), num_processes=self.num_processes)

    def setState(self, state: str) -> None:
        self.currently_active_key[-1] = state
        self.change_key(self.currently_active_key)
"""
    
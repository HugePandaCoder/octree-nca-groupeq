
import numpy as np
import collections

from batchgenerators.dataloading.data_loader import default_collate
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


def get_batchgenerators_dataloader_dataset(dataset_class, augmentations:bool, num_steps_per_epoch: int,
                                           batch_size: int):
    class BatchgeneratorsDataLoaderDataset(dataset_class):
        #class BatchgeneratorsDataLoaderDatasetDataSetIterator(collections.Iterator):
        #    pass





        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if augmentations:
                assert False, "Not implemented yet"
            else:
                self.transform = None

            self.internal_index = 0
            self.multithreaded_augmentor = MultiThreadedAugmenter(self, transform=None, num_processes=8)

        def setState(self, state: str) -> None:
            super().setState(state)
            self.reinitialize()

        def setPaths(self, images_path: str, images_list: str, labels_path: str, labels_list: str):
            super().setPaths(images_path, images_list, labels_path, labels_list)
            self.reinitialize()

        def reinitialize(self):
            self.internal_index = 0
            self.multithreaded_augmentor.restart()
            assert False, "reinitialize dataloader queues"

        def __iter__(self):
            return self.multithreaded_augmentor

        def __next__(self):
            if self.state == 'test':
                item = self[self.internal_index]
                self.internal_index += 1
                return default_collate([item])
            
            if self.internal_index >= num_steps_per_epoch:
                self.internal_index = 0
                raise StopIteration
            self.internal_index += 1

            batch_indices = np.random.randint(0, len(self), size=batch_size)

            batch = [self[i] for i in batch_indices]
            return default_collate(batch)
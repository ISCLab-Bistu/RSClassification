import os.path as osp
from os import PathLike

from .base_dataset import BaseDataset
from .builder import DATASETS


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


# Cross-validation was performed using KFoldDataset, which is available in dataset_wrappers.py
@DATASETS.register_module()
class RamanSpectral(BaseDataset):
    # raman type
    CLASSES = ['Health', 'Ovarian']

    def __len__(self):
        return len(self.data_infos['labels'])

    def __getitem__(self, idx):
        data = self.data_infos['spectrum'][idx:idx + 1, :]

        if self.test_mode:
            results = {'spectrum': data}
        else:
            label = self.data_infos['labels'][idx]
            results = {'spectrum': data, 'labels': label}
        return results

    # load data
    def load_annotations(self):
        results = {'raman_path': self.file_path, 'data_size': self.data_size, 'seed': self.seed}
        results = self.pipeline(results)

        return results

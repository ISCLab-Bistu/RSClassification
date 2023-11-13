# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import PIPELINES
import pandas as pd

from ..rampy import load_txt


@PIPELINES.register_module()
class LoadDataFromFile(object):
    def __init__(self, file_path=None, data_size=None):
        self.file_path = file_path
        self.data_size = data_size

    def __call__(self, results):
        print("LoadData")
        if self.file_path is None:
            if isinstance(results, str):
                self.file_path = results
            else:
                self.file_path = results['raman_path']

        if self.data_size is None:
            self.data_size = results['data_size']

        if self.data_size is not None:
            assert len(self.data_size) == 2
        if 'seed' not in results:
            results['seed'] = 2
        data = load_csv(self.file_path, self.data_size, results['seed'])
        results = {'labels': data[0], 'classes': data[1], 'raman_shift': data[2], 'spectrum': data[3]}

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'file_path={self.file_path}, '
                    f'data_size={self.data_size})')
        return repr_str


def load_csv(path, data_size, random_seed):
    # Load the corresponding data according to the path
    base_dir = '../../' + path

    all_df = pd.read_csv(base_dir)

    # raman_shift
    raman_shift = all_df.iloc[0:1, 2:].values
    raman_shift = raman_shift.flatten()

    df = all_df.iloc[1:]  # contains type, label, spectrum

    if data_size is None:
        data = df
    else:
        # Get the category
        df = df.sample(frac=1, random_state=random_seed)
        list_type = df['labels'].unique()

        assert data_size[0] <= 1
        assert data_size[1] <= 1

        # Get the first corresponding dataframe
        type_df = df[df['labels'] == list_type[0]]
        length = len(type_df)
        start = int(data_size[0] * length)
        end = int(length * data_size[1])
        data = type_df.iloc[start:end]

        # Loop through subsequent dataframes and concatenate them together
        for i in range(len(list_type) - 1):
            type_df = df[df['labels'] == list_type[i + 1]]
            length = len(type_df)
            start = int(data_size[0] * length)
            end = int(length * data_size[1])
            data = pd.concat([data, type_df.iloc[start:end]], axis=0)

    labels = data['labels'].values
    raman_type = data['raman_type'].drop_duplicates().values
    spectrum = data.iloc[:, 2:].values

    results = [labels, raman_type, raman_shift, spectrum]

    return results

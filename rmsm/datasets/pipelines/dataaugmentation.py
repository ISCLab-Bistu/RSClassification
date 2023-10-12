import random

import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class AddNoise(object):
    def __init__(self, num=None, noise_std=0.03, **kwargs):
        self.kwargs = kwargs
        self.num_new_data = num
        self.noise_std = noise_std

    def __call__(self, results):
        labels = results['labels']
        spectrum = results['spectrum']
        num_new_data = len(labels)
        length_label = len(labels)
        if self.num_new_data is not None:
            num_new_data = self.num_new_data

        indices = list(range(length_label))
        random.shuffle(indices)

        result_spectrum = []
        result_label = []
        for i in range(length_label):
            result_spectrum.append(spectrum[i])
            result_label.append(labels[i])
        for i in range(num_new_data):
            if i >= length_label:
                k = indices[i - length_label]
            else:
                k = indices[i]
            original_spectrum = spectrum[k]
            noise_std = self.noise_std * np.std(original_spectrum)
            noise = np.random.normal(scale=noise_std, size=original_spectrum.shape)
            new_data = original_spectrum + noise
            result_spectrum.append(new_data)
            result_label.append(labels[k])

        result_spectrum = np.array(result_spectrum)
        result_label = np.array(result_label)
        results['spectrum'] = result_spectrum
        results['labels'] = result_label
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_new_data={self.num_new_data}, '
                    f'noise_std={self.noise_std})')
        return repr_str


@PIPELINES.register_module()
class MoveRaman(object):
    def __init__(self, num=None, max_shift=3, move_ranges=[(200, 400), (400, 500), (500, 600), (600, 650)], **kwargs):
        self.kwargs = kwargs
        self.num_new_data = num
        self.max_shift = max_shift
        # different intervals
        self.move_ranges = move_ranges

    def __call__(self, results):
        labels = results['labels']
        spectrum = results['spectrum']
        num_new_data = len(labels)
        length_label = len(labels)
        if self.num_new_data is not None:
            num_new_data = self.num_new_data

        indices = list(range(length_label))
        random.shuffle(indices)

        # The mobile Raman spectrum is considered in different intervals
        result_spectrum = []
        result_label = []
        for i in range(length_label):
            result_spectrum.append(spectrum[i])
            result_label.append(labels[i])
        for i in range(num_new_data):
            if i >= length_label:
                k = indices[i - length_label]
            else:
                k = indices[i]
            original_spectrum = spectrum[k]
            shifted_data = original_spectrum.copy()

            shift = np.random.randint(-self.max_shift, self.max_shift)
            # Randomly move the data points in the selected range left and right
            for start, end in self.move_ranges:
                shifted_data[start:end] = np.roll(shifted_data[start:end], shift, axis=0)

            result_spectrum.append(shifted_data)
            result_label.append(labels[k])

        result_spectrum = np.array(result_spectrum)
        result_label = np.array(result_label)
        results['spectrum'] = result_spectrum
        results['labels'] = result_label
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_new_data={self.num_new_data}, '
                    f'max_shift={self.max_shift}), '
                    f'move_ranges={self.move_ranges}')
        return repr_str


@PIPELINES.register_module()
class IntensityFactory(object):
    def __init__(self, num=None, **kwargs):
        self.kwargs = kwargs
        self.num_new_data = num

    def __call__(self, results):
        labels = results['labels']
        spectrum = results['spectrum']
        num_new_data = len(labels)
        length_label = len(labels)
        if self.num_new_data is not None:
            num_new_data = self.num_new_data

        indices = list(range(length_label))  # 0-length
        random.shuffle(indices)  # 

        result_spectrum = []
        result_label = []
        for i in range(length_label):
            result_spectrum.append(spectrum[i])
            result_label.append(labels[i])
        for i in range(num_new_data):
            if i >= length_label:
                k = indices[i - length_label]
            else:
                k = indices[i]
            original_spectrum = spectrum[k]
            intensity_factor = np.random.uniform(0.2, 2)
            new_data = original_spectrum * intensity_factor
            result_spectrum.append(new_data)
            result_label.append(labels[k])

        result_spectrum = np.array(result_spectrum)
        result_label = np.array(result_label)
        results['spectrum'] = result_spectrum
        results['labels'] = result_label
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_new_data={self.num_new_data})')
        return repr_str

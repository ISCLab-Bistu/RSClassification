from collections.abc import Sequence

from ..rampy import normalise
import torch
import numpy as np
import rmsm
from rmsm.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not rmsm.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@PIPELINES.register_module()
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class DataToFloatTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key]).float()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Normalize(object):
    def __init__(self, method='intensity'):
        self.method = method

    def __call__(self, results):
        # You need to loop over the results data
        data = []
        x = results['raman_shift']
        for i in range(len(results['labels'])):
            y = results['spectrum'][i]
            data.append(normalise(y=y, x=x, method=self.method))
        data = np.array(data)  # numpy
        results['spectrum'] = data

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'method={self.method})')
        return repr_str


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "ram" and "labels".

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``rmsm.DataContainer`` and collected in ``data[ram_metas]``.
            Default: ('filename', 'ori_shape', 'ram_shape', 'flip',
            'flip_direction', 'ram_norm_cfg')

    Returns:
        dict: The result dict contains the following keys

            - keys in ``self.keys``
            - ``ram_metas`` if available
    """

    def __init__(self,
                 keys,
                 meta_keys=('raman_path', 'spectrum', 'labels',
                            'raman_shift', 'classes')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        ram_meta = {}
        for key in self.meta_keys:
            if key in results:
                ram_meta[key] = results[key]
        data['ram_metas'] = DC(ram_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'

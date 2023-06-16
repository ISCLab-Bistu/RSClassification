# Copyright (c) OpenMMLab. All rights reserved.
from .loading import LoadDataFromFile
from .compose import Compose
from .formatting import (Normalize, to_tensor, DataToFloatTensor, ToTensor, Collect)
from .remove_baseline import RemoveBaseline
from .spectrum_formatting import Flipping, Resampling
from .remove_Noise import Smoothing
from .dataaugmentation import AddNoise, MoveRaman, IntensityFactory

__all__ = [
    'LoadDataFromFile', 'Compose', 'Normalize',
    'to_tensor', 'DataToFloatTensor', 'ToTensor',
    'Collect', 'RemoveBaseline', 'Flipping', 'Smoothing',
    'Resampling', 'AddNoise', 'MoveRaman', 'IntensityFactory'
]

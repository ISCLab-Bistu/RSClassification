# Copyright (c) OpenMMLab. All rights reserved.
from .inference import init_model
from .train import init_random_seed, set_random_seed, train_model
from .test import multi_gpu_test, single_gpu_test

__all__ = [
    'set_random_seed', 'train_model', 'init_model',
    'init_random_seed', 'multi_gpu_test', 'single_gpu_test'
]

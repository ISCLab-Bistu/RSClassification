# Copyright (c) OpenMMLab. All rights reserved.
from rmsm.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from .dataloader import IPUDataLoader
    from .hook_wrapper import IPUFp16OptimizerHook
    from .runner import IPUBaseRunner, IPUEpochBasedRunner, IPUIterBasedRunner
    from .utils import cfg2options
    __all__ = [
        'cfg2options', 'IPUFp16OptimizerHook',
        'IPUDataLoader', 'IPUBaseRunner', 'IPUEpochBasedRunner',
        'IPUIterBasedRunner'
    ]

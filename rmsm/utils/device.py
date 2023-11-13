# Copyright (c) OpenMMLab. All rights reserved.
import rmsm
import torch
from rmsm.utils import digit_version


def auto_select_device() -> str:
    rmsm_version = digit_version(rmsm.__version__)
    if rmsm_version >= digit_version('1.6.0'):
        from rmsm.device import get_device
        return get_device()
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .linear_head import LinearClsHead
from .stacked_head import StackedLinearClsHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'ConformerHead'
]

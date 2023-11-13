# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import BaseClassifier
from ..builder import CLASSIFIERS, build_backbone, build_neck, build_head
from ..utils.augment import Augments


@CLASSIFIERS.register_module()
class RamanClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 nas=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(RamanClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        # 
        # self.cls_loss = ClsLoss(loss=loss)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def extract_feat(self, spectrum, stage='neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(spectrum)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def forward_dummy(self, spectrum):
        """Used for computing network flops.

        See `mmclassificaiton/tools/analysis_tools/get_flops.py`
        """
        return self.extract_feat(spectrum, stage='pre_logits')

    # 
    def forward_train(self, spectrum, labels, **kwargs):
        if self.augments is not None:
            spectrum, labels = self.augments(spectrum, labels)

        x = self.extract_feat(spectrum)

        losses = dict()
        loss = self.head.forward_train(x, labels)

        losses.update(loss)

        return losses

    def simple_test(self, spectrum, spectrum_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(spectrum)
        res = self.head.simple_test(x, **kwargs)

        return res

# Copyright (c) OpenMMLab. All rights reserved.
from rmsm.cnn import MODELS as rmsm_MODELS
from rmsm.cnn.bricks.registry import ATTENTION as rmsm_ATTENTION
from rmsm.utils import Registry

MODELS = Registry('models', parent=rmsm_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS

ATTENTION = Registry('attention', parent=rmsm_ATTENTION)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)

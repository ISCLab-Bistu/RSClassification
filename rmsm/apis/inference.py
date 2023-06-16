# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import rmsm
import numpy as np
import torch
from rmsm.parallel import scatter
from rmsm.runner import load_checkpoint

from rmsm.datasets.pipelines import Compose
from rmsm.models import build_classifier, build_backbone


def init_model(config, checkpoint=None, device='cuda:0', options=None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`rmsm.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = rmsm.Config.fromfile(config)
    elif not isinstance(config, rmsm.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from rmsm.datasets import RamanSpectral
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use ramanspectral by default.')
            model.CLASSES = RamanSpectral.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model, spectrum):
    """Inference Raman data with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        spectrum (str/ndarray): The Raman filename or loaded Raman.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(spectrum, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadDataFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadDataFromFile'))
        data = {'raman_path': spectrum, 'data_size': None}
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadDataFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(spectrum=spectrum)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    spectrum_metas = [data['spectrum'], data['labels']]

    # forward the model
    with torch.no_grad():
        scores = model(spectrum_metas=spectrum_metas, return_loss=False)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


def show_result_pyplot(model,
                       spectrum,
                       result,
                       raman_shift=None,
                       title='result',
                       wait_time=0):
    """Visualize the classification results on the Raman.

    Args:
        model (nn.Module): The loaded classifier.
        spectrum (str or np.ndarray): Raman spectrum.
        result (list): The classification result.
        raman_shift: Raman shift
        title (str): Title of the pyplot figure.
            Defaults to 'result'.
        wait_time (int): How many seconds to display the raman.
            Defaults to 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        spectrum,
        result,
        raman_shift=raman_shift,
        win_name=title,
        wait_time=wait_time)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from functools import partial

import rmsm
import numpy as np
import torch
from rmsm.runner import load_checkpoint
from torch import nn

from rmsm.models import build_classifier

torch.manual_seed(3)


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, L) = input_shape
    rng = np.random.RandomState(0)
    spectrum = rng.rand(*input_shape)
    labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'spectrum': torch.FloatTensor(spectrum).requires_grad_(True),
        'labels': torch.LongTensor(labels),
    }
    return mm_inputs


def pytorch2torchscript(model: nn.Module, input_shape: tuple, output_file: str,
                        verify: bool):
    """Export Pytorch model to TorchScript model through torch.jit.trace and
    verify the outputs are same between Pytorch and TorchScript.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output
            TorchScript model.
        verify (bool): Whether compare the outputs between Pytorch
            and TorchScript through loading generated output_file.
    """
    model.cpu().eval()

    num_classes = model.backbone.num_classes
    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    spectrums = mm_inputs.pop('spectrum')
    # labels = mm_inputs.pop('labels').squeeze(0)

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, spectrum_metas={}, return_loss=False)

    with torch.no_grad():
        trace_model = torch.jit.trace(model, spectrums)
        save_dir, _ = osp.split(output_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        trace_model.save(output_file)
        print(f'Successfully exported TorchScript model: {output_file}')
    model.forward = origin_forward

    if verify:
        # load by torch.jit
        jit_model = torch.jit.load(output_file)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(spectrums, spectrum_metas={}, return_loss=False)[0]

        pytorch_result = pytorch_result.detach().numpy()

        # get jit output
        jit_result = jit_model(spectrums)[0].detach().numpy()

        if not np.allclose(pytorch_result, jit_result):
            raise ValueError(
                'The outputs are different between Pytorch and TorchScript')
        print('The outputs are same between Pytorch and TorchScript')


if __name__ == '__main__':
    # Input data dimension, 900 is Raman shift length
    input_shape = (1, 1, 900)

    # configuration file
    cfg = rmsm.Config.fromfile("../../configs/transformer/raman_covid-19.py")
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    load_checkpoint(classifier, './covid_trasformer.pth', map_location='cpu')

    # convert model to TorchScript file
    pytorch2torchscript(
        classifier,
        input_shape,
        output_file='covid_transformer.pt',
        verify=True)

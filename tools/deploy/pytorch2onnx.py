# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from functools import partial

import numpy as np
import onnxruntime as rt
import torch

import rmsm
from rmsm.models import build_classifier
from rmsm.onnx import register_extra_symbolics
from rmsm.runner import load_checkpoint

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


def pytorch2onnx(model,
                 input_shape,
                 opset_version=12,
                 dynamic_export=False,
                 show=False,
                 output_file='tmp.onnx',
                 dimension=False,
                 do_simplify=False,
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 12.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(model.backbone, 'num_classes', -1) > 0:
        num_classes = model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                             'backbone, please check the config file.')

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    spectrums = mm_inputs.pop('spectrum')
    labels = mm_inputs.pop('labels').squeeze(0)
    spectrum_list = {'spectrum': spectrums, 'return_loss': False, 'post_process': False}
    # spectrum_list = [spectrum[None, :] for spectrum in spectrums]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, spectrum_metas={})
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'length',
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}

    with torch.no_grad():
        torch.onnx.export(
            model,
            (spectrum_list,),
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=True,
            dynamic_axes=dynamic_axes,
            verbose=show,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')

        # Whether to add dimension information
        if dimension:
            import onnx
            model_file = output_file
            onnx_model = onnx.load(model_file)
            onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
    model.forward = origin_forward

    if do_simplify:
        import onnx
        import onnxsim
        from rmsm import digit_version

        min_required_version = '0.4.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnxsim>={min_required_version}'

        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            print('Failed to simplify ONNX model.')
    if verify:
        # check by onnx
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        if dynamic_export:
            dynamic_test_inputs = _demo_mm_inputs(
                (input_shape[0], input_shape[1], input_shape[2] * 2), model.backbone.num_classes)
            spectrum = mm_inputs.pop('spectrum')
            labels = mm_inputs.pop('labels')
            spectrum_list = [spectrum, labels]

        # check the numerical value
        # get pytorch output
        pytorch_result = model(spectrum_list, return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: spectrum_list[0].detach().numpy()})[0]
        print(onnx_result)
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


"""  
    We need to change the input shape and checkpoint
"""
if __name__ == '__main__':
    # Output data dimension，（batch_size, channel, raman_shift）,1480 is the Raman displacement length
    input_shape = (1, 1, 1480)

    cfg = rmsm.Config.fromfile(
        "../../configs/mobilenet/raman_ovarian_cancer.py")  # Load the configuration file path，Need to match the input data
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    # loaded model file
    checkpoint = './model/ovarian_mobile.pth'
    load_checkpoint(classifier, checkpoint, map_location='cpu')  #

    # print(classifier)

    # convert model to onnx file（output_file is the output of the onnx model）
    pytorch2onnx(
        classifier,
        input_shape,
        output_file='./model/MobileNetV2.onnx',
        opset_version=12,
        show=False,
        dynamic_export=True,
        dimension=True,
        do_simplify=True,
        verify=False)

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

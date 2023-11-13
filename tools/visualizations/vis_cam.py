# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import pkg_resources
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils import get_2d_projection
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm

import rmsm.datasets.rampy as rp
from rmsm import Config
from rmsm import digit_version
from rmsm.apis import init_model
from rmsm.datasets.pipelines import Compose
from rmsm.utils import to_1tuple

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# set of transforms, which just change data format, not change the pictures
FORMAT_TRANSFORMS_SET = {'Normalize', 'DataToFloatTensor', 'ToTensor', 'Collect'}


def plot_curve(raman_shift, spectrum):
    x = raman_shift
    y = spectrum
    # y[y < 0] = 0

    # smooth
    y_smooth = rp.smooth(x, y, method="whittaker", Lambda=10 ** 0.5)

    # baseline
    roi = np.array([[-29, 4090]])
    ycalc_poly, base_poly = rp.baseline(x, y_smooth, roi, 'als', lam=10 ** 5, p=0.05)

    # normalization
    y_norm = rp.normalise(y=ycalc_poly, x=x, method="minmax")

    return x, y_norm


class MMActivationsAndGradients(ActivationsAndGradients):
    """Activations and gradients manager for rmsm models."""

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(
            x, return_loss=False, softmax=False, post_process=False)


class My_GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            My_GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        length = input_tensor.size(-1)
        return 1, length

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # return grads
        return np.mean(grads, axis=2, keepdims=True)

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)

        weighted_activations = weights * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=0)

        return cam


# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': My_GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def apply_transforms(spectrum_path, pipeline_cfg):
    """Apply transforms pipeline and get both formatted data and the spectrum
    without formatting."""
    data = {'raman_path': spectrum_path, 'data_size': None}

    def split_pipeline_cfg(pipeline_cfg):
        """to split the transfoms into spectrum_transforms and
        format_transforms."""
        spectrum_transforms_cfg, format_transforms_cfg = [], []
        if pipeline_cfg[0]['type'] != 'LoadDataFromFile':
            pipeline_cfg.insert(0, dict(type='LoadDataFromFile'))
        for transform in pipeline_cfg:
            if transform['type'] in FORMAT_TRANSFORMS_SET:
                format_transforms_cfg.append(transform)
            else:
                spectrum_transforms_cfg.append(transform)
        return spectrum_transforms_cfg, format_transforms_cfg

    spectrum_transforms, format_transforms = split_pipeline_cfg(pipeline_cfg)
    spectrum_transforms = Compose(spectrum_transforms)
    format_transforms = Compose(format_transforms)

    intermediate_data = spectrum_transforms(data)
    inference_spectrum = copy.deepcopy(intermediate_data['spectrum'])
    format_data = format_transforms(intermediate_data)

    return format_data, inference_spectrum


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object."""

    print(method.lower())
    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    # Release the original hooks in ActivationsAndGradients to use
    # MMActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = MMActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    cur_layer = model
    layer_names = layer_str.strip().split('.')

    def get_children_by_name(model, name):
        try:
            return getattr(model, name)
        except AttributeError as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    def get_children_by_eval(model, name):
        try:
            return eval(f'model{name}', {}, {'model': model})
        except (AttributeError, IndexError) as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    for layer_name in layer_names:
        match_res = re.match('(?P<name>.+?)(?P<indices>(\\[.+\\])+)',
                             layer_name)
        if match_res:
            layer_name = match_res.groupdict()['name']
            indices = match_res.groupdict()['indices']
            cur_layer = get_children_by_name(cur_layer, layer_name)
            cur_layer = get_children_by_eval(cur_layer, indices)
        else:
            cur_layer = get_children_by_name(cur_layer, layer_name)

    return cur_layer


def get_default_traget_layers(model, args):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = []
    for m in model.backbone.modules():
        if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d)):
            norm_layers.append(m)
    if len(norm_layers) == 0:
        raise ValueError(
            '`--target-layers` is empty. Please use `--preview-model`'
            ' to check keys at first and then specify `target-layers`.')
    # if the model is CNN model or Swin model, just use the last norm
    # layer as the target-layer, if the model is ViT model, the final
    # classification is done on the class token computed in the last
    # attention block, the output will not be affected by the 14x14
    # channels in the last layer. The gradient of the output with
    # respect to them, will be 0! here use the last 3rd norm layer.
    # means the first norm of the last decoder block.
    if args.vit_like:
        if args.num_extra_tokens:
            num_extra_tokens = args.num_extra_tokens
        elif hasattr(model.backbone, 'num_extra_tokens'):
            num_extra_tokens = model.backbone.num_extra_tokens
        else:
            raise AttributeError('Please set num_extra_tokens in backbone'
                                 " or using 'num-extra-tokens'")

        # if a vit-like backbone's num_extra_tokens bigger than 0, view it
        # as a VisionTransformer backbone, eg. DeiT, T2T-ViT.
        if num_extra_tokens >= 1:
            print('Automatically choose the last norm layer before the '
                  'final attention block as target_layer..')
            return [norm_layers[-3]]
    print('Automatically choose the last norm layer as target_layer.')
    target_layers = [norm_layers[-1]]
    return target_layers


'''
Currently only grad-cam visualization is supported, and the steps are as follows:
1. Firstly, modify the configuration file cam_config to configure grad-cam visualization;
2, by observing the grad-cam gradient map, find out the values with relatively high gradient values
3. Intercept part of grad-cam values (modify start and end in the code) to draw visual heat map and spectrogram
4. Save the generated image

In order to use this method, you need to modify the cam_config.py, 'start' and 'end'.
'''


def main():
    # Get some parameters by loading a configuration file
    args = Config.fromfile('./cam_config.py')
    cfg = Config.fromfile(args.config)

    cfg.device = 'cuda'
    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    # apply transform and perpare data
    data_spectrum, src_spectrum = apply_transforms(args.spectrum, cfg.data.test.pipeline)

    spectrum = data_spectrum['spectrum']
    raman_shift = data_spectrum['raman_shift']
    spectrum_size = len(raman_shift)
    print(spectrum_size)

    # build target layers
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_traget_layers(model, args)

    # init a cam grad calculator
    use_cuda = ('cuda' in args.device)
    # reshape_transform = build_reshape_transform(model, args)
    cam = init_cam(args.method, model, target_layers, use_cuda,
                   reshape_transform=None)

    # warp the target_category with ClassifierOutputTarget in grad_cam>=1.3.7,
    # to fix the bug in #654.
    targets = None
    if args.target_category:
        grad_cam_v = pkg_resources.get_distribution('grad_cam').version
        if digit_version(grad_cam_v) >= digit_version('1.3.7'):
            from pytorch_grad_cam.utils.model_targets import \
                ClassifierOutputTarget
            targets = [ClassifierOutputTarget(c) for c in args.target_category]
        else:
            targets = args.target_category

    # calculate cam grads and show|save the visualization image
    grayscale_cam = cam(
        spectrum.unsqueeze(0),
        targets,
        eigen_smooth=args.eigen_smooth,
        aug_smooth=args.aug_smooth)

    print(grayscale_cam)
    grayscale_cam = grayscale_cam - 0.1
    # grayscale_cam = 1 - grayscale_cam
    grayscale_cam = grayscale_cam.reshape(spectrum_size)

    grayscale_cam[grayscale_cam < 0] = 0
    grayscale_cam[grayscale_cam > 1] = 1

    # start and end are the intercepted spectral ranges
    start = 199
    end = 800
    str_list = ['Human Respiratory Virus', 'Ovarian cancer', 'Cancer cell culture medium',
                'Hematopoietic Stem cells']
    index = 0

    # x = raman_shift.reshape(spectrum_size)[start:end]
    # y = spectrum.reshape(spectrum_size)[start:end]
    df = pd.read_csv("../../" + args.spectrum)
    labels = ['labels']
    df.drop(labels, axis=1, inplace=True)  #

    x = df.iloc[0:1, start + 1:end + 1].values
    x = x.flatten()
    # spectrum
    y = df.iloc[1:2, start + 1:end + 1].values
    y = y.flatten()
    # print(x)

    x, y = plot_curve(x, y)

    plt.figure(figsize=(9, 6), dpi=500)  #
    plt.rc('font', family='Times New Roman')
    #
    heatmap = grayscale_cam[start:end]
    heatmap = np.flip(heatmap)  #
    x = np.flip(x)
    y = np.flip(y)

    # with open("save_results/grad_txt/" + cfg.work_dir + ".txt", mode="w", encoding='utf-8') as file:
    #     for i in range(len(heatmap)):
    #         save_str = str(heatmap[i])
    #         write_f = save_str + "\n"
    #         file.write(write_f)

    # Generate the heatmap data
    heatmap = np.uint8(255 * heatmap)
    heatmap_results = []
    for i in range(len(x)):
        heatmap_results.append(heatmap)
    heatmap = np.array(heatmap_results)

    plt.plot(x, y, 'b')
    heatmap = plt.imshow(heatmap, cmap="autumn_r", aspect='auto', alpha=0.7, extent=[x[0], x[len(x) - 1], 0, 1])
    # Add a color bar
    heatmap.set_clim(vmin=0, vmax=250)
    plt.colorbar(heatmap)
    # Set the image title and label
    plt.tick_params(labelsize=16)  #
    plt.title('Tag: ' + str_list[index], fontsize=18)
    plt.xlabel("Raman shift cm$^{-1}$", fontsize=18)
    plt.ylabel("Normalized intensity", fontsize=18)
    plt.xticks(np.arange(1000, 1601, 100))
    plt.savefig('./save_results/tif/' + cfg.work_dir + '.tif')  #
    plt.show()

    # Filter Raman shifts by printing gradient values and viewing grad-cam gradient maps
    raman_shift = raman_shift.reshape(spectrum_size)
    # print(1 - grayscale_cam)
    plt.plot(raman_shift, grayscale_cam, color='red')

    plt.show()


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Sequence

import rmsm
import numpy as np
import torch
import torch.distributed as dist
from rmsm.runner import BaseModule, auto_fp16
import matplotlib

matplotlib.use('Agg')  # Do not show pictures，Save the corresponding picture directly

import matplotlib.pyplot as plt


class BaseClassifier(BaseModule, metaclass=ABCMeta):
    """Base class for classifiers."""

    def __init__(self, init_cfg=None):
        super(BaseClassifier, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def extract_feat(self, spectrums, stage=None):
        pass

    def extract_feats(self, spectrums, stage=None):
        assert isinstance(spectrums, Sequence)
        kwargs = {} if stage is None else {'stage': stage}
        for spectrum in spectrums:
            yield self.extract_feat(spectrum, **kwargs)

    @abstractmethod
    def forward_train(self, spectrums, **kwargs):
        """
        Args:
            spectrum (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    @abstractmethod
    def simple_test(self, spectrum, **kwargs):
        pass

    def forward_test(self, spectrum, **kwargs):
        """
        Args:
            spectrums (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all raman data in the batch.
        """
        return self.simple_test(spectrum, **kwargs)

    @auto_fp16(apply_to=('spectrum', ))
    def forward(self, spectrum, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, spectrum and spectrum_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, spectrum and spectrum_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(spectrum, **kwargs)
        else:
            return self.forward_test(spectrum, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)  # invokeforward
        loss, log_vars = self._parse_losses(losses)  # Format the result

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['spectrum'][0]))

        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['spectrum'][0]))

        return outputs

    # Present the resulting model,
    def show_result(self,
                    spectrum,
                    result=None,
                    raman_shift=None,
                    win_name='result',
                    out_file=None):
        """Draw `result` over `spectrum`.

        Args:
            spectrum: Raman spectrum(array)
            result: Raman spectral classification results
            raman_shift: Raman frequency shift
            win_name:Spectrum data Title
            out_file:Spectrum data saving path
        Returns:
            spectrum (ndarray): spectrum data results
        """
        # 
        batch, num = spectrum.shape
        print(batch, num)
        pred_score = result['pred_score']
        pred_label = result['pred_label']
        pred_class = result['pred_class']
        if raman_shift is None:
            raman_shift = np.arange(0, num, 1)
        # plot raman spectrum
        for i in range(batch):
            plt.plot(raman_shift, spectrum[i])
            plt.title(win_name, fontsize=12, fontweight="bold")
            plt.xlabel("Raman shift, cm$^{-1}$", fontsize=12)
            plt.ylabel("Normalized intensity, a. u.", fontsize=12)
            ax = plt.gca()
            if result:
                result_str = str(pred_class[i]) + "：" + str(pred_score[i])
                plt.text(0.5, 0.9, result_str, fontsize=8, verticalalignment="top",
                         horizontalalignment="right", transform=ax.transAxes)
            if out_file:
                plt.draw()
                plt.savefig(out_file + 'pic-{}.png'.format(i + 1))
                plt.pause(1)
            plt.close()

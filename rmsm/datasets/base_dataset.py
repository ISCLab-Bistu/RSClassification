# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import List

import rmsm
import numpy as np
import torch
from torch.utils.data import Dataset

from rmsm.core.evaluation import precision_recall_f1, calculate_confusion_matrix
from rmsm.models.losses import accuracy
from .pipelines import Compose


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `rmsm.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self,
                 file_path,
                 pipeline,
                 data_size=0.3,
                 classes=None,
                 test_mode=False):
        super(BaseDataset, self).__init__()
        self.file_path = expanduser(file_path)
        self.pipeline = Compose(pipeline)
        self.data_size = data_size
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()
        self.CLASSES = self.get_classes(classes)

    @abstractmethod
    def load_annotations(self):
        pass

    # Update the self.data infos based on the pipeline
    def update_pipeline(self, up_pipeline, indices):
        up_pipeline = Compose(up_pipeline)
        self.data_infos['labels'] = self.data_infos['labels'][indices]
        self.data_infos['spectrum'] = self.data_infos['spectrum'][indices]
        self.data_infos = up_pipeline(self.data_infos)

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all data.
        """
        labels = [data for data in self.data_infos['labels']]
        labels = np.vstack(labels)
        return labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Data category of specified index.
        """

        return [int(self.data_infos['labels'][idx])]

    def get_classes(self, classes=None):
        if classes is None:
            return self.data_infos['classes']

        if isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def __len__(self):
        return len(self.data_infos['labels'])

    def __getitem__(self, idx):
        data = self.data_infos['spectrum'][idx]
        label = self.data_infos['labels'][idx]
        return data, label

    def evaluate(self,
                 results,
                 metric=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1,)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support', 'confusion'
        ]
        eval_results = {}
        results = np.vstack(results)
        labels = self.get_labels()

        if indices is not None:
            labels = labels[indices]
        num_rams = len(results)
        assert len(labels) == num_rams, 'dataset testing results should ' \
                                        'be of the same length as labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1,))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        # Calculate confusion matrix
        if 'confusion' in metrics:
            confusion_matrix = calculate_confusion_matrix(results, labels)
            eval_results['confusion'] = confusion_matrix

        return eval_results

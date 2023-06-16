# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from rmsm.parallel import collate
from rmsm.runner import get_dist_info
from rmsm.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
SAMPLERS = Registry('sampler')


# Build data set
def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset, KFoldDataset)

    if cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            separate_eval=cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'KFoldDataset':
        cp_cfg = copy.deepcopy(cfg)
        if cp_cfg.get('test_mode', None) is None:
            cp_cfg['test_mode'] = (default_args or {}).pop('test_mode', False)
        if cp_cfg['dataset'].get('data_size', None) is not None:
            cp_cfg['dataset']['data_size'] = None
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'], default_args)
        cp_cfg.pop('type')
        dataset = KFoldDataset(**cp_cfg)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


# build dataloader
def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     round_up=True,
                     seed=None,
                     pin_memory=True,
                     persistent_workers=True,
                     sampler_cfg=None,
                     **kwargs):
    rank, world_size = get_dist_info()

    # Custom sampler logic
    if sampler_cfg:
        # shuffle=False when val and test
        sampler_cfg.update(shuffle=shuffle)
        sampler = build_sampler(
            sampler_cfg,
            default_args=dict(
                dataset=dataset, num_replicas=world_size, rank=rank,
                seed=seed))
    # Default sampler logic
    elif dist:
        sampler = build_sampler(
            dict(
                type='DistributedSampler',
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                round_up=round_up,
                seed=seed))
    else:
        sampler = None

    # If sampler exists, turn off dataloader shuffle
    if sampler is not None:
        shuffle = False

    if dist:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    else:
        return build_from_cfg(cfg, SAMPLERS, default_args=default_args)

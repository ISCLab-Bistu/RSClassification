# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import time

import torch
import torch.distributed as dist

import rmsm
from rmsm import __version__
from rmsm.apis import init_random_seed, set_random_seed, train_model
from rmsm.datasets import build_dataset
from rmsm.models import build_classifier
from rmsm.runner import get_dist_info
from rmsm.utils import (auto_select_device, get_root_logger,
                        setup_multi_processes, collect_env)


def main(path):
    # Get the corresponding config file after parsing
    cfg = rmsm.Config.fromfile(path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set multi-process settings
    setup_multi_processes(cfg)

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(path))[0])

    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # create work_dir, logs, checkpoint
    rmsm.mkdir_or_exist(osp.abspath(cfg.work_dir))
    rmsm.mkdir_or_exist(osp.abspath(cfg.work_dir + "/checkpoint"))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(path)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # Log recorder
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if cfg.get('device', None) is None:
        cfg.device = auto_select_device()
        # cfg.device = 'cpu'
    seed = init_random_seed(cfg.get('seed'), device=cfg.device)
    seed = seed + dist.get_rank() if cfg.get('diff_seed') else seed
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed)
    cfg.seed = seed
    meta['seed'] = seed

    # Initialize the backbone network and weights
    model = build_classifier(cfg.model)
    model.init_weights()

    # build train_dataset
    datasets = [build_dataset(cfg.data.train)]

    # workflow represents the training flow (verification set, need to add test_mode=true)
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    # save rmsm version, config file content and class names in
    # runner as meta data
    meta.update(
        dict(
            rmsm_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES))

    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        validate=True,
        timestamp=timestamp,
        device=cfg.device,
        meta=meta)


if __name__ == '__main__':
    path = "../../configs/resnet/raman_ovarian_cancer.py"
    main(path)

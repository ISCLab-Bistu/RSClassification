# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
from datetime import datetime
from pathlib import Path

import rmsm
import torch
from rmsm import Config, DictAction
from rmsm.runner import get_dist_info, init_dist

from rmsm import __version__
from rmsm.apis import init_random_seed, set_random_seed, train_model
from rmsm.datasets import build_dataset
from rmsm.models import build_classifier
from rmsm.utils import collect_env, get_root_logger, load_json_log

TEST_METRICS = ('precision', 'recall', 'f1_score', 'support', 'CP',
                'CR', 'CF1', 'OP', 'OR', 'OF1', 'accuracy')

prog_description = """K-Fold cross-validation. 

To start a 5-fold cross-validation experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5

To resume a 5-fold cross-validation from an interrupted experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --resume-from work_dirs/fold2/latest.pth

To summarize a 5-fold cross-validation:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --summary
"""  # noqa: E501


def copy_config(old_cfg):
    """deepcopy a Config object."""
    new_cfg = Config()
    _cfg_dict = copy.deepcopy(old_cfg._cfg_dict)
    _filename = copy.deepcopy(old_cfg._filename)
    _text = copy.deepcopy(old_cfg._text)
    super(Config, new_cfg).__setattr__('_cfg_dict', _cfg_dict)
    super(Config, new_cfg).__setattr__('_filename', _filename)
    super(Config, new_cfg).__setattr__('_text', _text)
    return new_cfg


def train_single_fold(cfg, fold, seed, distributed=False, test_data=True):
    # create the work_dir for the fold
    work_dir = osp.join(cfg.work_dir, f'fold{fold}')
    cfg.work_dir = work_dir

    print(cfg.work_dir)
    # create work_dir
    rmsm.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # wrap the dataset cfg
    train_load = cfg.data.train.pipeline.pop(0)
    test_load = cfg.data.test.pipeline.pop(0)
    val_load = cfg.data.val.pipeline.pop(0)
    train_pipeline = cfg.data.train.pipeline
    test_pipeline = cfg.data.test.pipeline
    cfg.data.train.pipeline = [train_load]
    cfg.data.test.pipeline = [test_load]
    cfg.data.val.pipeline = [val_load]

    train_dataset = dict(
        type='KFoldDataset',
        fold=fold,
        dataset=cfg.data.train,
        num_splits=cfg.num_splits,
        seed=seed,
        test_mode='train',
        pipeline=train_pipeline,
    )
    val_dataset = dict(
        type='KFoldDataset',
        fold=fold,
        # Use the same dataset with training.
        dataset=cfg.data.val,
        num_splits=cfg.num_splits,
        seed=seed,
        test_mode='val',
        pipeline=train_pipeline,
    )
    test_dataset = dict(
        type='KFoldDataset',
        fold=fold,
        dataset=cfg.data.test,
        num_splits=cfg.num_splits,
        seed=seed,
        test_mode='test',
        pipeline=test_pipeline,
    )
    cfg.data.train = train_dataset
    cfg.data.val = val_dataset
    cfg.data.test = test_dataset

    # dump config
    stem, suffix = osp.basename(cfg.config_path).rsplit('.', 1)
    cfg.dump(osp.join(cfg.work_dir, f'{stem}_fold{fold}.{suffix}'))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
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
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(
        f'-------- Cross-validation: [{fold + 1}/{cfg.num_splits}] -------- ')

    # set random seeds
    # Use different seed in different folds
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset['dataset'] = cfg.data.train['dataset']
        datasets.append(build_dataset(val_dataset))

    # Enable or nottesttest
    if test_data:
        test_datasets = build_dataset(test_dataset)
    meta.update(
        dict(
            rmsm_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            kfold=dict(fold=fold, num_splits=cfg.num_splits)))
    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        test_dataset=test_datasets,
        validate=True,
        distributed=distributed,
        timestamp=timestamp,
        device='cuda',
        meta=meta)


def summary(cfg):
    summary = dict()
    for fold in range(cfg.num_splits):
        work_dir = Path(cfg.work_dir) / f'fold{fold}'

        # Find the latest training log
        log_files = list(work_dir.glob('*.log.json'))
        if len(log_files) == 0:
            continue
        log_file = sorted(log_files)[-1]

        date = datetime.fromtimestamp(log_file.lstat().st_mtime)
        summary[fold] = {'date': date.strftime('%Y-%m-%d %H:%M:%S')}

        # Find the latest eval log
        json_log = load_json_log(log_file)
        epochs = sorted(list(json_log.keys()))
        eval_log = {}

        def is_metric_key(key):
            for metric in TEST_METRICS:
                if metric in key:
                    return True
            return False

        for epoch in epochs[::-1]:
            if any(is_metric_key(k) for k in json_log[epoch].keys()):
                eval_log = json_log[epoch]
                break

        summary[fold]['epoch'] = epoch
        summary[fold]['metric'] = {
            k: v[0]  # the value is a list with only one item.
            for k, v in eval_log.items() if is_metric_key(k)
        }
    show_summary(cfg, summary)


def show_summary(cfg, summary_data):
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        raise ImportError('Please run `pip install rich` to install '
                          'package `rich` to draw the table.')

    console = Console()
    table = Table(title=f'{cfg.num_splits}-fold Cross-validation Summary')
    table.add_column('Fold')
    metrics = summary_data[0]['metric'].keys()
    for metric in metrics:
        table.add_column(metric)
    table.add_column('Epoch')
    table.add_column('Date')

    for fold in range(cfg.num_splits):
        row = [f'{fold + 1}']
        if fold not in summary_data:
            table.add_row(*row)
            continue
        for metric in metrics:
            metric_value = summary_data[fold]['metric'].get(metric, '')

            def format_value(value):
                if isinstance(value, float):
                    return f'{value:.2f}'
                if isinstance(value, (list, tuple)):
                    return str([format_value(i) for i in value])
                else:
                    return str(value)

            row.append(format_value(metric_value))
        row.append(str(summary_data[fold]['epoch']))
        row.append(summary_data[fold]['date'])
        table.add_row(*row)

    console.print(table)


def main(config_path):
    cfg = Config.fromfile(config_path)  # The corresponding configuration file is obtained after parsing
    cfg.config_path = config_path

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(config_path))[0])

    # if args.summary:
    #     summary(args, cfg)
    #     return

    # resume from the previous experiment
    if cfg.get('resume_from', None) is not None:
        resume_kfold = torch.load(cfg.resume_from).get('meta',
                                                       {}).get('kfold', None)
        if resume_kfold is None:
            raise RuntimeError(
                'No "meta" key in checkpoints or no "kfold" in the meta dict. '
                'Please check if the resume checkpoint from a k-fold '
                'cross-valid experiment.')
        resume_fold = resume_kfold['fold']
    else:
        resume_fold = 0

    # init distributed env first, since logger depends on the dist info.
    if cfg.get('launcher') == 'none':
        distributed = False
    else:
        distributed = True
        # init_dist(cfg.get('launcher'))
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # init a unified random seed
    seed = init_random_seed(cfg.get('seed'))

    # create work_dir
    rmsm.mkdir_or_exist(osp.abspath(cfg.work_dir))

    cfg.num_splits = 10
    num_splits = 10  # Several fold cross verification
    folds = range(resume_fold, num_splits)

    for fold in folds:
        cfg_ = copy_config(cfg)
        if fold != resume_fold:
            cfg_.resume_from = None
        train_single_fold(cfg_, fold, seed)

    summary(cfg)


if __name__ == '__main__':
    main("../../configs/efficientnet/raman_single_cell.py")
    # main("../../configs/resnet/raman_covid-19.py")
    # cfg = Config.fromfile("../../configs/mobilenet/raman_covid-19.py")
    # cfg.num_splits = 10
    # summary(cfg)

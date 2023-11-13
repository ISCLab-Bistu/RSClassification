# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
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
    work_dir = osp.join(cfg.work_dir, f'{cfg.search_train}')
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
        pipeline=test_pipeline,
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


def main(cfg):
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

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
    seed = 626124256

    # create work_dir
    rmsm.mkdir_or_exist(osp.abspath(cfg.work_dir))

    cfg.num_splits = 10
    num_splits = 10  # Several fold cross verification
    folds = range(resume_fold, num_splits)
    folds = [0]

    for fold in folds:
        cfg_ = copy_config(cfg)
        if fold != resume_fold:
            cfg_.resume_from = None
        train_single_fold(cfg_, fold, seed)


if __name__ == '__main__':
    config_path = '../../configs/transformer/raman_single_cell.py'
    # Configuring the transformer's search Space
    search_d_model = [128, 256, 512]
    search_N_e = [1, 2, 4]
    search_heads = [2, 4, 8]

    for i in range(len(search_d_model)):
        for j in range(len(search_N_e)):
            for k in range(len(search_heads)):
                # Get the corresponding config file after parsing
                cfg = rmsm.Config.fromfile(config_path)
                cfg.model.backbone['d_model'] = search_d_model[i]
                cfg.model.backbone['N_e'] = search_N_e[j]
                cfg.model.backbone['heads'] = search_heads[k]

                # work_dir is determined in this priority: CLI > segment in file > filename
                if cfg.get('work_dir', None) is None:
                    # use config filename as default work_dir if cfg.work_dir is None
                    cfg.work_dir = osp.join('./work_dirs',
                                            osp.splitext(osp.basename(config_path))[0])

                cfg.config_path = config_path
                cfg.work_dir = cfg.work_dir + "_backbone"
                cfg.search_train = "d_" + str(search_d_model[i]) + "_N_" + str(search_N_e[j]) + "_H_" + str(search_heads[k])
                main(cfg)

# if __name__ == '__main__':
#     config_path = '../../configs/transformer/raman_single_cell.py'
#     # Configuring the transformer's search Space
#     search_d_model = [128, 256, 512]
#     search_N_e = [1, 2, 4]
#     search_heads = [2, 4, 8]
#
#     for i in range(len(search_d_model)):
#         # Get the corresponding config file after parsing
#         cfg = rmsm.Config.fromfile(config_path)
#         cfg.model.backbone['d_model'] = search_d_model[i]
#
#         # work_dir is determined in this priority: CLI > segment in file > filename
#         if cfg.get('work_dir', None) is None:
#             # use config filename as default work_dir if cfg.work_dir is None
#             cfg.work_dir = osp.join('./work_dirs',
#                                     osp.splitext(osp.basename(config_path))[0])
#
#         cfg.config_path = config_path
#         cfg.work_dir = cfg.work_dir + "_d_model"
#         cfg.search_train = search_d_model[i]
#         main(cfg)
#
#     # The f1_score is compared and the best parameter is obtained
#     compare_filepath = osp.join(cfg.work_dir, f'{"show_dir"}')
#     compare_file = os.listdir(compare_filepath)
#
#     best_f1 = 0
#     search_best_d_model = 0
#     for file in compare_file:
#         open_file = osp.join(compare_filepath, f'{file}')
#         with open(open_file, 'r') as f:
#             data = json.load(f)
#             search_para = file[8:11]
#             f1 = data['f1_score']
#             if best_f1 < f1:
#                 best_f1 = f1
#                 search_best_d_model = search_para
#
#     # Obtain the optimal parameters in each step
#     for i in range(len(search_N_e)):
#         # Get the corresponding config file after parsing
#         cfg = rmsm.Config.fromfile(config_path)
#         cfg.model.backbone['d_model'] = int(search_best_d_model)
#         cfg.model.backbone['N_e'] = search_N_e[i]
#
#         # work_dir is determined in this priority: CLI > segment in file > filename
#         if cfg.get('work_dir', None) is None:
#             # use config filename as default work_dir if cfg.work_dir is None
#             cfg.work_dir = osp.join('./work_dirs',
#                                     osp.splitext(osp.basename(config_path))[0])
#
#         cfg.config_path = config_path
#         cfg.work_dir = cfg.work_dir + "_N_e"
#         cfg.search_train = search_N_e[i]
#         main(cfg)
#
#     # The f1_score is compared and the best parameter is obtained
#     compare_filepath = osp.join(cfg.work_dir, f'{"show_dir"}')
#     compare_file = os.listdir(compare_filepath)
#
#     best_f1 = 0
#     search_best_d_model = 0
#     for file in compare_file:
#         open_file = osp.join(compare_filepath, f'{file}')
#         with open(open_file, 'r') as f:
#             data = json.load(f)
#             search_para = file[8:11]
#             f1 = data['f1_score']
#             if best_f1 < f1:
#                 best_f1 = f1
#                 search_best_d_model = search_para
#
#     for i in range(len(search_heads)):
#         # Get the corresponding config file after parsing
#         cfg = rmsm.Config.fromfile(config_path)
#         cfg.model.backbone['d_model'] = int(search_best_d_model)
#         cfg.model.backbone['N_e'] = search_N_e[i]
#         cfg.model.backbone['heads'] = search_heads[i]
#
#         # work_dir is determined in this priority: CLI > segment in file > filename
#         if cfg.get('work_dir', None) is None:
#             # use config filename as default work_dir if cfg.work_dir is None
#             cfg.work_dir = osp.join('./work_dirs',
#                                     osp.splitext(osp.basename(config_path))[0])
#
#         cfg.config_path = config_path
#         cfg.work_dir = cfg.work_dir + "_N_e"
#         cfg.search_train = search_N_e[i]
#         main(cfg)

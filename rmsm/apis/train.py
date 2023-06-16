# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import random
import warnings
from numbers import Number

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import random_split

import rmsm
from rmsm.core import (DistEvalHook, EvalHook, DistOptimizerHook)
from rmsm.datasets import build_dataloader, build_dataset
from rmsm.runner import (DistSamplerSeedHook, Fp16OptimizerHook,
                         build_optimizer, get_dist_info, build_runner, wrap_fp16_model, load_checkpoint)
from rmsm.utils import get_root_logger, wrap_distributed_model, wrap_non_distributed_model


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                test_dataset=None,
                validate=False,
                distributed=False,
                timestamp=None,
                device=None,
                meta=None):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.get('ipu_replicas') if device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    # trian_loaders and val_loaders
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    cfg['device'] = device

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = wrap_distributed_model(
            model,
            cfg.device,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = wrap_non_distributed_model(
            model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # build runner
    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config,
            loss_scale=fp16_cfg['loss_scale'],
            distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed and cfg.runner['type'] == 'EpochBasedRunner':
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val)
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            'drop_last': False,  # Not drop last by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/rmsm/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow)

    # The final results should be tested and saved after the training
    if test_dataset:
        from rmsm.apis import single_gpu_test, multi_gpu_test
        test_dataset_length = len(test_dataset)
        # build the dataloader
        # The default loader config
        loader_cfg = dict(
            # cfg.gpus will be ignored if distributed
            num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
        )

        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader'
            ]
        })

        test_loader_cfg = {**loader_cfg, 'shuffle': False, 'sampler_cfg': None, 'samples_per_gpu': test_dataset_length,
                           **cfg.data.get('test_dataloader', {})}
        # the extra round_up data will be removed during gpu/cpu collect
        data_loader = build_dataloader(test_dataset, **test_loader_cfg)

        # build the model and load checkpoint
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint_path = cfg.work_dir + '/latest.pth'  # use latest.pth

        # use the best.pth
        # checkpoint_file = os.listdir(cfg.work_dir)
        # for file in checkpoint_file:
        #     if 'best' in file:
        #         checkpoint_path = cfg.work_dir + '/' + file

        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cuda')

        if 'CLASSES' in checkpoint.get('meta', {}):
            CLASSES = checkpoint['meta']['CLASSES']
        else:
            from rmsm.datasets import RamanSpectral
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use ramanspectrum by default.')
            CLASSES = RamanSpectral.CLASSES

        if not distributed:
            model = wrap_non_distributed_model(
                model, device=cfg.device, device_ids=cfg.gpu_ids)
            model.CLASSES = CLASSES
            show_kwargs = {}
            outputs = single_gpu_test(model, data_loader, False, cfg.work_dir + "/show_dir",
                                      **show_kwargs)
        else:
            model = wrap_distributed_model(
                model,
                device=cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, cfg.tmpdir,
                                     cfg.gpu_collect)

        rank, _ = get_dist_info()
        print(cfg.work_dir)
        # 
        word_list = cfg.work_dir.split('\\')
        args_out = word_list[0] + "/show_dir/results_" + word_list[len(word_list) - 1] + ".json"
        args_out_items = 'all'
        if rank == 0:
            results = {}
            logger = get_root_logger()

            eval_results = test_dataset.evaluate(
                results=outputs,
                logger=logger)
            results.update(eval_results)
            for k, v in eval_results.items():
                # if isinstance(v, np.ndarray):
                #     v = [round(out, 2) for out in v.tolist()]
                # elif isinstance(v, Number):
                #     v = round(v, 2)
                # else:
                #     raise ValueError(f'Unsupport metric type: {type(v)}')
                print(f'\n{k} : {v}')

            if args_out:
                if 'none' not in args_out_items:
                    scores = np.vstack(outputs)
                    pred_score = np.max(scores, axis=1)
                    # pred_label = np.argmax(scores, axis=1)
                    # pred_class = [CLASSES[lb] for lb in pred_label]
                    res_items = {
                        'pred_score': pred_score,
                        # 'pred_label': pred_label,
                        # 'pred_class': pred_class
                    }
                    if 'all' in args_out_items:
                        results.update(res_items)
                    else:
                        for key in args_out_items:
                            results[key] = res_items[key]
                print(f'\ndumping results to {args_out}')
                rmsm.utils.dump(results, args_out)

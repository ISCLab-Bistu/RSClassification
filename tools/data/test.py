# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from numbers import Number

import rmsm
import numpy as np
import torch
from rmsm import Config

from rmsm.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from rmsm.apis import multi_gpu_test, single_gpu_test
from rmsm.datasets import build_dataloader, build_dataset
from rmsm.models import build_classifier
from rmsm.utils import (auto_select_device, get_root_logger,
                        setup_multi_processes, wrap_distributed_model,
                        wrap_non_distributed_model)


def main(path):
    # Get the corresponding config file after parsing
    cfg = Config.fromfile(path)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # Set the gpu id in the config file
    if cfg.get('gpu_ids') is None:
        cfg.gpu_ids = [0]
    cfg.device = auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    distributed = False
    # init_dist('pytorch')

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
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
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint_path = cfg.work_dir + '/latest.pth'
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
    args_out = cfg.work_dir + "./show_dir/results.json"
    args_out_items = 'all'
    if rank == 0:
        results = {}
        logger = get_root_logger()

        eval_results = dataset.evaluate(
            results=outputs,
            logger=logger)
        results.update(eval_results)
        for k, v in eval_results.items():
            print(f'\n{k} : {v}')

        if args_out:
            if 'none' not in args_out_items:
                scores = np.vstack(outputs)
                pred_score = np.max(scores, axis=1)
                pred_label = np.argmax(scores, axis=1)
                pred_class = [CLASSES[lb] for lb in pred_label]
                res_items = {
                    # 'class_scores': scores,
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class
                }
                if 'all' in args_out_items:
                    results.update(res_items)
                else:
                    for key in args_out_items:
                        results[key] = res_items[key]
            print(f'\ndumping results to {args_out}')
            rmsm.utils.dump(results, args_out)


if __name__ == '__main__':
    path = "../../configs/resnet/raman_ovarian_cancer.py"
    main(path)

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

import rmsm
from rmsm.apis import init_random_seed, set_random_seed
from rmsm.datasets import build_dataset, build_dataloader
from rmsm.models import build_nas, build_loss
from rmsm.models.nas.uitls import log_info_config
from rmsm.runner import get_dist_info
from rmsm.utils import (auto_select_device, get_root_logger,
                        setup_multi_processes, collect_env)

best_genotype = "Genotype(normal=[[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], [('sep_conv_3x3', 2), ('dil_conv_5x5', 0)], [('dil_conv_5x5', 3), ('dil_conv_5x5', 0)], [('dil_conv_3x3', 4), ('dil_conv_5x5', 2)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 0), ('max_pool_3x3', 1)], [('skip_connect', 1), ('dil_conv_5x5', 0)], [('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], [('max_pool_3x3', 1), ('max_pool_3x3', 2)]], reduce_concat=range(2, 6))"


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

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, "tb"))

    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    # create work_dir, logs
    rmsm.mkdir_or_exist(osp.abspath(cfg.work_dir))
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
    seed = init_random_seed(cfg.get('seed'), device=cfg.device)
    seed = seed + dist.get_rank() if cfg.get('diff_seed') else seed
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed)
    cfg.seed = seed
    meta['seed'] = seed

    # Initialize the backbone network and weights
    use_aux = cfg.aux_weight > 0.
    cfg.model.auxiliary = use_aux

    criterion = build_loss(cfg.loss)
    cfg.model.genotype = best_genotype
    model = build_nas(cfg.model)
    model.init_weights()
    model = nn.DataParallel(model, device_ids=cfg.gpu_ids).to(cfg.device)
    model.to(cfg.device)

    # model size
    mb_params = log_info_config.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # build train_dataset
    train_dataset = build_dataset(cfg.data.train)

    # build optimizer(weights and alphas)
    optimizer = torch.optim.SGD(model.parameters(), cfg.w_optimizer.w_lr, momentum=cfg.w_optimizer.w_momentum,
                                weight_decay=cfg.w_optimizer.w_weight_decay)

    # The default loader config
    device = cfg.device
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.get('ipu_replicas') if device == 'ipu' else len(cfg.gpu_ids),
        dist=False,
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
    # build data_loader
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    train_loader = build_dataloader(train_dataset, **train_loader_cfg)

    # The specific dataloader settings
    cfg.data.val.test_mode = False
    val_dataset = build_dataset(cfg.data.val)
    val_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        'drop_last': False,  # Not drop last by default
        **cfg.data.get('val_dataloader', {}),
    }
    valid_loader = build_dataloader(val_dataset, **val_loader_cfg)

    # Cosine annealing algorithm
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

    best_top1 = 0.
    # training loop
    for epoch in range(cfg.epochs):
        lr_scheduler.step()
        drop_prob = cfg.drop_path_prob * epoch / cfg.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train_augment(train_loader, model, optimizer, criterion, epoch, logger=logger, writer=writer, cfg=cfg)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate_augment(valid_loader, model, criterion, epoch, cur_step, logger=logger, writer=writer, cfg=cfg)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        log_info_config.save_checkpoint(model, cfg.work_dir, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train_augment(train_loader, model, optimizer, criterion, epoch, logger, writer, cfg):
    top1 = log_info_config.AverageMeter()
    losses = log_info_config.AverageMeter()

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, train_results in enumerate(train_loader):
        X = train_results['spectrum']
        y = train_results['labels']
        X, y = X.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if cfg.aux_weight > 0.:
            loss += cfg.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        prec1, prec5 = log_info_config.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1) ({top1.avg:.1%})".format(
                    epoch + 1, cfg.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, cfg.epochs, top1.avg))


def validate_augment(valid_loader, model, criterion, epoch, cur_step, logger, writer, cfg):
    top1 = log_info_config.AverageMeter()
    losses = log_info_config.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, val_results in enumerate(valid_loader):
            X = val_results['spectrum']
            y = val_results['labels']
            X, y = X.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = log_info_config.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)

            if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1) ({top1.avg:.1%})".format(
                        epoch + 1, cfg.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, cfg.epochs, top1.avg))

    return top1.avg


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, logger, writer, cfg):
    top1 = log_info_config.AverageMeter()
    losses = log_info_config.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, (train_results, val_results) in enumerate(zip(train_loader, valid_loader)):
        trn_X = train_results['spectrum']
        trn_y = train_results['labels']
        trn_X, trn_y = trn_X.to(cfg.device, non_blocking=True), trn_y.to(cfg.device, non_blocking=True)

        val_X = val_results['spectrum']
        val_y = val_results['labels']
        val_X, val_y = val_X.to(cfg.device, non_blocking=True), val_y.to(cfg.device, non_blocking=True)
        N = trn_X.size(0)

        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.weights(), cfg.w_optimizer.w_grad_clip)
        w_optim.step()

        prec1, prec5 = log_info_config.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)

        if step % cfg.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1) ({top1.avg:.1%})".format(
                    epoch + 1, cfg.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, cfg.epochs, top1.avg))


def validate(valid_loader, model, epoch, logger, cur_step, writer, cfg):
    top1 = log_info_config.AverageMeter()
    losses = log_info_config.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, val_results in enumerate(valid_loader):
            X = val_results['spectrum']
            y = val_results['labels']
            X, y = X.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
            N = X.size(0)
            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = log_info_config.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)

            if step % cfg.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1) ({top1.avg:.1%})".format(
                        epoch + 1, cfg.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, cfg.epochs, top1.avg))

    return top1.avg


if __name__ == '__main__':
    path = "../../configs/nas/single_augment_darts.py"
    main(path)

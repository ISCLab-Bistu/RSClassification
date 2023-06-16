# Copyright (c) OpenMMLab. All rights reserved.


def wrap_non_distributed_model(model, device='cuda', dim=0, *args, **kwargs):
    """Wrap module in non-distributed environment by device type.

    - For CUDA, wrap as :obj:`rmsm.parallel.MMDataParallel`.
    - For CPU & IPU, not wrap the model.

    Args:
        model(:class:`nn.Module`): model to be parallelized.
        device(str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim(int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        model(nn.Module): the model to be parallelized.
    """
    if device == 'cuda':
        from rmsm.parallel import MMDataParallel
        model = MMDataParallel(model.cuda(), dim=dim, *args, **kwargs)
    elif device == 'cpu':
        model = model.cpu()
    elif device == 'ipu':
        model = model.cpu()
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model


def wrap_distributed_model(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    - For CUDA, wrap as :obj:`rmsm.parallel.MMDistributedDataParallel`.
    - Other device types are not supported by now.

    Args:
        model(:class:`nn.Module`): module to be parallelized.
        device(str): device type, mlu or cuda.

    Returns:
        model(:class:`nn.Module`): the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
               DistributedDataParallel.html
    """
    if device == 'cuda':
        from rmsm.parallel import MMDistributedDataParallel
        model = MMDistributedDataParallel(model.cuda(), *args, **kwargs)
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model


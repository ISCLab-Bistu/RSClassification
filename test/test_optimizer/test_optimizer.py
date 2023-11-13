# -*- coding: utf-8 -*-
import torch.nn as nn
from rmsm.runner.optimizer.builder import OPTIMIZERS

cfg = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

model = nn.Conv2d(1, 1, 1)
cfg['params'] = model.parameters()
optimizer = OPTIMIZERS.build(cfg)
print(optimizer)

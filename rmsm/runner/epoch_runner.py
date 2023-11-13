# -*- coding: utf-8 -*-
import logging

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from time import time
import os.path as osp
import torch
import torch.nn as nn
from os.path import join
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union)

import rmsm
from .builder import RUNNERS

# Fine-tune CNN for first fold
best_val = 0
no_improvement = 0
max_no_improvement = 5


@RUNNERS.register_module()
class EpochRunner(object):
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function=torch.nn.CrossEntropyLoss(),
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None) -> None:

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        # create work_dir
        if isinstance(work_dir, str):
            self.work_dir: Optional[str] = osp.abspath(work_dir)
            rmsm.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._max_epochs = max_epochs

    def train(self, data_loaders):
        self.model.train()
        running_loss = 0
        running_acc = 0
        trainLoader = data_loaders[0]
        loader_len = len(trainLoader)
        for step, data_batch in enumerate(trainLoader):
            # data = data.to(device)  GPU
            data = data_batch[0]
            label = data_batch[1]
            data = data.float()
            label = label.float()
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss_function(outputs, label.long())
            # outputs = self.model.train_step(data_batch, self.optimizer)
            # loss = outputs['loss']
            loss.backward()
            self.optimizer.step()

            # lossbatchSize
            running_loss += loss.item()
            _, predict = torch.max(outputs, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.item() / len(label)

        # 
        running_loss /= loader_len
        running_acc /= loader_len

        return running_loss, running_acc

    @torch.no_grad()
    def val(self, data_loaders):
        self.model.eval()
        running_loss = 0
        running_acc = 0
        valLoader = data_loaders[1]
        loader_len = len(valLoader)
        for step, data_batch in enumerate(valLoader):
            data = data_batch[0]
            label = data_batch[1]
            data = data.float()
            label = label.float()
            outputs = self.model(data)
            loss = self.loss_function(outputs, label.long())
            print(loss)
            # outputs = self.model.val_step(data_batch, self.optimizer)
            # loss = outputs['loss']
            # lost
            running_loss += loss.item()
            _, predict = torch.max(outputs, 1)
            num_correct = (predict == label).sum()
            running_acc += num_correct.item() / len(label)

        # 
        running_loss /= loader_len
        running_acc /= loader_len
        return running_loss, running_acc

    def run(self, data_loaders: List[DataLoader], workflow):
        # train
        # log_writer = SummaryWriter(self.work_dir + '/runs')

        output_path = self.work_dir + '\checkpoint'
        best_loss = 1000
        for epoch in range(self._max_epochs):
            train_loss, train_acc = self.train(data_loaders)
            # lr_scheduler.step()
            eval_loss, eval_acc = self.val(data_loaders)
            # log_writer.add_scalar('Loss/train', float(train_loss), epoch)  # loss
            # log_writer.add_scalar('Loss/eval', float(eval_loss), epoch)  # loss
            print("[%d/%d] Loss: %.5f, Acc: %.2f" % (epoch+1, self._max_epochs, train_loss, 100 * train_acc))
            print("Test: Loss: %.5f, Acc: %.2f %%" % (eval_loss, 100 * eval_acc))
            if eval_loss < best_loss:
                save_path = join(output_path, 'chechpoint.pt')
                torch.save(self.model.state_dict(), save_path, _use_new_zipfile_serialization=True)
                best_loss = eval_loss
                print("save best model")

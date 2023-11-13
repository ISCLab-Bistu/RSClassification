# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from resnet import ResNet, ResNetV1c, ResNetV1d
from resnet50 import ResNet50
from mobilenet_v2 import MobileNetV2
from mobilenet_test import MobileNetV2_test
from resnet50_test import ResNetTest

from rmsm.models.backbones.resnet import ResNet

input_data = torch.FloatTensor(64, 1, 1480)

classifier = ResNet()
classifier(input_data)

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

search_kernel_size = [3, 5, 7]
search_stride = [2, 3, 4]
search_padding = [1, 2]


@BACKBONES.register_module()
class SearchAlexNet(BaseBackbone):
    """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super(SearchAlexNet, self).__init__()
        self.num_classes = num_classes
        # self.features = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=3, stride=2),
        #
        #     nn.Conv1d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=3, stride=2),
        #
        #     nn.Conv1d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool1d(kernel_size=3, stride=2),
        # )

        # Each model is implemented using a combination of hyperparameters
        self.model1 = self.build_module(1, 64, search_kernel_size, search_stride, search_padding)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        # model2
        self.model2 = self.build_module(64, 192, search_kernel_size, search_stride, search_padding)
        self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        # model3
        self.model3 = self.build_module(192, 384, search_kernel_size, search_stride, search_padding)
        # self.max_pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.model4 = self.build_module(384, 256, search_kernel_size, search_stride, search_padding)
        # self.max_pool4 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.model5 = self.build_module(256, 256, search_kernel_size, search_stride, search_padding)
        self.max_pool5 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(6)
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def build_module(self, input_channel, out_channel, search_kernel_size, search_stride, search_padding):
        model = []
        for i in range(len(search_kernel_size)):
            for j in range(len(search_stride)):
                for k in range(len(search_padding)):
                    module = nn.Sequential(
                        nn.Conv1d(input_channel, out_channel, kernel_size=search_kernel_size[i],
                                  stride=search_stride[j],
                                  padding=search_padding[k]),
                        nn.ReLU(inplace=True),
                    )
                    module.to('cuda')
                    model.append(module)
        return model

    def oprational(self, model, input):
        x_model = [item(input) for item in model]
        for i in range(len(x_model)):
            print(x_model[i].shape)
        # Convert alpha_shape to a PyTorch tensor
        x_model = torch.stack(x_model)
        alpha = nn.Parameter(torch.randn(len(x_model), requires_grad=True))
        print(alpha.shape)

        out = sum(alpha[i] * feature for i, feature in enumerate(x_model))

        return out

    def forward(self, x):
        # x = self.features(x)
        # Each network is trained
        x = self.oprational(model=self.model1, input=x)
        x = self.max_pool1(x)
        x = self.oprational(model=self.model2, input=x)
        x = self.max_pool2(x)
        x = self.oprational(model=self.model3, input=x)
        x = self.oprational(model=self.model4, input=x)
        x = self.oprational(model=self.model5, input=x)
        x = self.max_pool5(x)

        x = self.avgpool(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6)
            x = self.classifier(x)

        return (x,)

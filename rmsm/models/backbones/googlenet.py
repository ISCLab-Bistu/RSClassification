import torch.nn as nn
import torch as torch
from ..builder import BACKBONES


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3_reduce, out_channels_3x3,
                 out_channels_5x5_reduce, out_channels_5x5, out_channels_pool):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv1d(in_channels, out_channels_1x1, kernel_size=1)

        # 1x1 convolution -> 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_3x3_reduce, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv1d(out_channels_3x3_reduce, out_channels_3x3, kernel_size=3, padding=1)
        )

        # 1x1 convolution -> 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_5x5_reduce, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Conv1d(out_channels_5x5_reduce, out_channels_5x5, kernel_size=5, padding=2)
        )

        # 3x3 max pooling -> 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1x1_output = self.branch1x1(x)
        branch3x3_output = self.branch3x3(x)
        branch5x5_output = self.branch5x5(x)
        branch_pool_output = self.branch_pool(x)

        outputs = [branch1x1_output, branch3x3_output, branch5x5_output, branch_pool_output]
        return torch.cat(outputs, 1)


# googlenet converges slowly and requires many rounds of training
@BACKBONES.register_module()
class GoogLeNet(nn.Module):
    def __init__(self, input_dim=815, num_classes=3):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.a3 = InceptionModule(64, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        # Rest of the Inception modules...

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(480, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        # Rest of the forward pass...
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

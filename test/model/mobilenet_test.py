import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * t, 1),
            nn.BatchNorm1d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv1d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm1d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv1d(in_channels * t, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2_test(nn.Module):

    def __init__(self, class_num=3):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv1d(1, 32, 1),
            # nn.Conv2d(1, 32, 1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv1d(320, 1280, 1),
            # nn.Conv2d(320, 1280, 1),
            nn.BatchNorm1d(1280),
            nn.ReLU6(inplace=True)
        )

        # self.conv2 = nn.Conv2d(1280, class_num, 1)

        self.linear1 = nn.Linear(131, 1)
        self.linear2 = nn.Linear(1280, 6)
        self.linear_ac = nn.ReLU()
        self.BN = nn.BatchNorm1d(1044)
        self.Dropout0 = nn.Dropout(0.3)
        self.Dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        print(x.shape)
        x = self.pre(x)
        print(x.shape)
        x = self.stage1(x)
        print(x.shape)
        x = self.stage2(x)
        print(x.shape)
        x = self.stage3(x)
        print(x.shape)
        x = self.stage4(x)
        print(x.shape)
        x = self.stage5(x)
        print(x.shape)
        x = self.stage6(x)
        print(x.shape)
        x = self.stage7(x)
        print(x.shape)
        x = self.conv1(x)
        x = self.linear1(x)
        x = self.Dropout0(x)
        x = x.reshape(-1, 1280)
        x = self.linear2(x)
        x = self.Dropout1(x)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

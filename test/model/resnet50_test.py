import torch
import torch.nn as nn


# todo Bottleneck
class Bottleneck(nn.Module):
    """
    __init__
        in_channel：Residual block input channel number
        out_channel：Residual block output channel number
        stride：Convolution step size
        downsample：in_make_layerAssign a value to a function，shortcut H/2 W/2
    """
    expansion = 4  # 3Channel expansion ratio of each convolution layer

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1,
                               bias=False)  # H,Winvariability。C: in_channel -> out_channel
        self.bn1 = nn.BatchNorm1d(num_features=out_channel)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               bias=False, padding=1)  # H/2，W/2。Cinvariability
        self.bn2 = nn.BatchNorm1d(num_features=out_channel)
        self.conv3 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1,
                               stride=1, bias=False)  # H,Winvariability。C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm1d(num_features=out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x  # Temporarily store the original input asshortcut
        if self.downsample is not None:
            identity = self.downsample(
                x)  # ，shortcut:H/2，W/2。C: out_channel -> 4*out_channel(ResNetdownsample)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 
        out = self.relu(out)

        return out


# todo ResNet
class ResNetTest(nn.Module):
    """
    __init__
        block: 
        block_num: ,list,resnet50=[3,4,6,3]
        num_classes: 

    _make_layer
        block: 
        channel: stage，resnet50:64,128,256,512
        block_num: stageblock
        stride: 
    """

    def __init__(self, block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=3):
        super(ResNetTest, self).__init__()
        self.in_channel = 64  # conv1

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.in_channel, kernel_size=7, stride=1, padding=3,
                               bias=False)  # H/2,W/2。C:3->64
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # H/2,W/2。C
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0],
                                       stride=1)  # H,W。downsampleshortcut，out_channel=64x4=256
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1],
                                       stride=2)  # H/2, W/2。downsampleshortcut，out_channel=128x4=512
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2],
                                       stride=2)  # H/2, W/2。downsampleshortcut，out_channel=256x4=1024
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3],
                                       stride=2)  # H/2, W/2。downsampleshortcut，out_channel=512x4=2048

        self.avgpool = nn.AdaptiveAvgPool1d(1)  # ->(1,1)，=
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        for m in self.modules():  # 
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  # shorcut
        if stride != 1 or self.in_channel != channel * block.expansion:  # resnet50：conv2H,W/2，x4，shortcutx4。conv3,4,5，H,W/2，shortcutx4
            downsample = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channel, out_channels=channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),  # out_channelsx4，strideH,W/2
                nn.BatchNorm1d(num_features=channel * block.expansion))

        layers = []  # convi_xlayers，i={2,3,4,5}
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample,
                            stride=stride))  # convi_x，downsamplestride
        self.in_channel = channel * block.expansion  # _make_layer，self.in_channelx4

        for _ in range(1, block_num):  # (block_num-1)
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)  # '*'list

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

"""
Modified ResNet in pytorch

Uses 3x3 conv in first layer instead of 7x7
See our supplementary for exact details.

Original paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

from typing import TYPE_CHECKING

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


class BasicBlock(nn.Module):
    """
    Basic Block for ResNet-18 and ResNet-34

    :param in_channels: input channels
    :type in_channels: int
    :param out_channels: output channels
    :type out_channels: int
    :param stride: the stride of the first block of this layer
    :type stride: int
    """

    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_dropout = False):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1, inplace=True)

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        out = self.residual_function(x)
        if self.use_dropout:
            out = self.dropout(out)
        return nn.ReLU(inplace=True)(out + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    Residual block for ResNet-50+ layers

    :param in_channels: input channels
    :type in_channels: int
    :param out_channels: output channels
    :type out_channels: int
    :param stride: the stride of the first block of this layer
    :type stride: int
    """

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_dropout = False):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BottleNeck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1, inplace=True)

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        out = self.residual_function(x)
        if self.use_dropout:
            out = self.dropout(out)
        return nn.ReLU(inplace=True)(out + self.shortcut(x))


class ResNet(nn.Module):
    """
    Modified ResNet in pytorch

    Uses 3x3 conv in first layer instead of 7x7
    See our supplementary for exact details.

    Original paper: Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385v1

    :param block: Block type, Basic or Bottleneck
    :type block: Union[BasicBlock, BottleNeck]
    :param num_block: Block no's.
    :type num_block: List[int]
    :param num_classes: No of output labels.
    :type num_classes: int
    :param small_dense_density: Equivalent parameter density of Small-Dense model
    :type small_dense_density: float
    :param zero_init_residual: Whether to init batchnorm gamma to 0.
        Empirically achieves better performance, Improved residual training.
        Default 0.
    :type zero_init_residual: bool
    """

    def __init__(
        self,
        block: "Union[BasicBlock, BottleNeck]",
        num_block: "List[int]",
        num_classes: int = 100,
        small_dense_density: float = 1.0,
        zero_init_residual: bool = True,
        use_dropout = False
    ):
        super().__init__()

        self.use_dropout = use_dropout

        small_dense_density = np.sqrt(small_dense_density)

        self.in_channels = int(64 * small_dense_density)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, int(64 * small_dense_density), kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(int(64 * small_dense_density)),
            nn.ReLU(inplace=True),
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(
            block, int(64 * small_dense_density), num_block[0], 1, self.use_dropout
        )
        self.conv3_x = self._make_layer(
            block, int(128 * small_dense_density), num_block[1], 2, self.use_dropout
        )
        self.conv4_x = self._make_layer(
            block, int(256 * small_dense_density), num_block[2], 2, self.use_dropout
        )
        self.conv5_x = self._make_layer(
            block, int(512 * small_dense_density), num_block[3], 2, self.use_dropout
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            int(512 * small_dense_density) * block.expansion, num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck) or isinstance(m, BasicBlock):
                    nn.init.constant_(m.residual_function[-1].weight, 0)

    def _make_layer(
        self,
        block: "Union[BasicBlock, BottleNeck]",
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_dropout
    ) -> nn.Sequential:
        """
        Make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block.

        :param block: Block type, Basic or Bottleneck
        :type block: Union[BasicBlock, BottleNeck]
        :param out_channels: output depth channel number of this layer
        :type out_channels: int
        :param num_blocks: how many blocks per layer
        :type num_blocks: int
        :param stride: the stride of the first block of this layer
        :type stride: int
        :return: return a resnet layer
        :rtype: nn.Sequential
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_dropout))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return F.log_softmax(output, dim=1)


def resnet18():
    """
    return a ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], use_dropout = False)


def resnet34():
    """
    return a ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], use_dropout = False)


def resnet50():
    """
    return a ResNet-50 model
    """
    #model = ResNet(BottleNeck, [3, 4, 6, 3], use_dropout = False)
    return ResNet(BottleNeck, [3, 4, 6, 3], use_dropout = False)


def resnet101():
    """
    return a ResNet-101 model
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], use_dropout = False)


def resnet152():
    """
    return a ResNet-152 model
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], use_dropout = False)


if __name__ == "__main__":
    from torchsummary import summary

    resnet50 = ResNet(BottleNeck, [3, 4, 6, 3], 10, use_dropout = False)
    #resnet50 = ResNet(BottleNeck, [3, 4, 6, 3], 10)
    #resnet50 = ResNet(BasicBlock, [3, 4, 6, 3], 100)
    summary(resnet50, (3, 32, 32))

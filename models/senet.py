import torch.nn as nn
from models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck


class SELayer(nn.Module):  # Has fixed number of FC layers.
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_features=num_channels // reduction_ratio, out_features=num_channels),
            nn.Sigmoid()
        )

    def forward(self, tensor):
        # Global average pooling. Remove dimensions for feeding to linear.
        features = tensor.mean(dim=(-2, -1), keepdim=False)
        channel_attention = self.layer(features).view(tensor.shape[0], tensor.shape[1], 1, 1).expand_as(tensor)
        # Return scale layer.
        return tensor * channel_attention


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, reduction_ratio=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SEBasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes, reduction_ratio)  # Scale Layer.
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # Apply channel-wise attention.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, reduction_ratio=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.se = SELayer(planes * self.expansion, reduction_ratio)  # Scale Layer.
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)  # Apply channel-wise attention.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet(nn.Module):

    def __init__(self, blocks, num_layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):

        super().__init__()

        if issubclass(blocks, nn.Module):
            assert hasattr(blocks, 'expansion'), 'All blocks must have an "expansion" attribute.'
            blocks = [blocks] * 4
        else:
            assert len(blocks) == 4, 'This should be a list of 4 block types.'
            for block in blocks:
                assert issubclass(block, nn.Module), 'Elements must be Pytorch Modules.'
                assert hasattr(block, 'expansion'), 'All blocks must have an "expansion" attribute.'

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Modified to allow multiple block types.
        self.layer1 = self._make_layer(blocks[0], 64, num_layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(blocks[1], 128, num_layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(blocks[2], 256, num_layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(blocks[3], 512, num_layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocks[3].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (Bottleneck, SEBottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, (BasicBlock, SEBasicBlock)):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        # The first block of the layer has downsampling.
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def se_resnet50_cifar100():
    return SEResNet(SEBottleneck, num_layers=[3, 4, 6, 3], num_classes=100)

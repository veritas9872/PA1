import torch
import torch.nn as nn
from models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
from models.bam import dil_conv1x1, dil_conv3x3

"""
Idea: Try different spatial attention module. 
Channel attention seems to be fine with CBAM style with both max pooling and avg pooling.

Plan for new spatial attention:

1. Do max-pooling on the incoming data to reduce feature size and select only the most important points.
2. Do a 1x1 conv on the results and reduce channel number by reduction factor. This will squeeze information.
3. Use dilated 3x3 conv to increase receptive field even further. Reduce channel number to (pool_stride ** 2).
4. Use pixel shuffle to expand the (pool_stride ** 2) channels to a single channel.
"""


class CALayer(nn.Module):
    """
    Channel Attention Layer for my module.
    """
    mine = True

    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))  # Global Maximum Pooling

        self.layer = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_features=num_channels // reduction_ratio, out_features=num_channels)
        )

    def forward(self, tensor):
        shape = tensor.shape
        # Remove dimensions for feeding to linear.
        gap = self.gap(tensor).view(shape[0], shape[1])
        gmp = self.gmp(tensor).view(shape[0], shape[1])

        # Concat on batch axis for parallel calculation on FC layer.
        features = self.layer(torch.cat((gap, gmp), dim=0))
        features = features[:shape[0], ...] + features[shape[0]:, ...]  # Split along batch dimension.
        channel_attention = features.view(shape[0], shape[1], 1, 1).expand_as(tensor)
        return channel_attention


class SALayer(nn.Module):
    """
    Spatial Attention Layer for my module.
    """
    mine = True

    def __init__(self, num_channels, reduction_ratio=16, dilation_value=2, pool_stride=2):
        super().__init__()

        num_reduced = num_channels // reduction_ratio
        self.layer = nn.Sequential(
            nn.MaxPool2d(pool_stride),  # Only using maxpool here because this is from the front, not from the side.

            dil_conv1x1(in_planes=num_channels, out_planes=num_reduced, bias=True),
            nn.ReLU(),

            dil_conv3x3(in_planes=num_reduced, out_planes=pool_stride ** 2, dilation=dilation_value, bias=True),
            nn.PixelShuffle(pool_stride)
        )

    def forward(self, tensor):
        spatial_attention = self.layer(tensor).expand_as(tensor)
        return spatial_attention


class MyLayer(nn.Module):
    """
    My attention layer
    """
    mine = True

    def __init__(self, num_channels, reduction_ratio=16, dilation_value=4, pool_stride=2, use_ca=True, use_sa=True):
        super().__init__()
        self.channel_attention = CALayer(num_channels=num_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SALayer(num_channels, reduction_ratio, dilation_value, pool_stride)
        self.sigmoid = nn.Sigmoid()

        self.use_ca = use_ca
        self.use_sa = use_sa

    def forward(self, tensor):

        if self.use_ca and self.use_sa:
            attention = self.sigmoid(self.channel_attention(tensor) + self.spatial_attention(tensor))

        elif not (self.use_ca or self.use_sa):
            attention = 1  # Do nothing

        elif self.use_ca and not self.use_sa:
            attention = self.sigmoid(self.channel_attention(tensor))

        elif not self.use_ca and self.use_sa:
            attention = self.sigmoid(self.spatial_attention(tensor))

        else:
            raise ValueError('Impossible combination')

        return tensor * attention


class MyBasicBlock(nn.Module):
    expansion = 1
    mine = True

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction_ratio=16, dilation_value=2, pool_stride=2, use_ca=True, use_sa=True):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.att = MyLayer(num_channels=planes, reduction_ratio=reduction_ratio,
                           dilation_value=dilation_value, pool_stride=pool_stride, use_ca=use_ca, use_sa=use_sa)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.att(out)  # No if conditions necessary. Att is for attention.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyBottleneck(nn.Module):
    expansion = 4
    mine = True

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 reduction_ratio=16, dilation_value=2, pool_stride=2, use_ca=True, use_sa=True):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.att = MyLayer(num_channels=planes * self.expansion, reduction_ratio=reduction_ratio,
                           dilation_value=dilation_value, pool_stride=pool_stride, use_ca=use_ca, use_sa=use_sa)
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
        out = self.att(out)  # Added attention module here.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyResNet(nn.Module):
    mine = True

    def __init__(self, block, num_layers, num_classes=100, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 reduction_ratio=16, dilation_value=2, pool_stride=2, use_ca=True, use_sa=True):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        options = dict(reduction_ratio=reduction_ratio, dilation_value=dilation_value,
                       pool_stride=pool_stride, use_ca=use_ca, use_sa=use_sa)
        self.layer1 = self._make_layer(block, 64, num_layers[0], **options)
        self.layer2 = self._make_layer(block, 128, num_layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], **options)
        self.layer3 = self._make_layer(block, 256, num_layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], **options)
        self.layer4 = self._make_layer(block, 512, num_layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], **options)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                if isinstance(m, (Bottleneck, MyBottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, (BasicBlock, MyBasicBlock)):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False,
                    reduction_ratio=16, dilation_value=2, pool_stride=2, use_ca=True, use_sa=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        if hasattr(block, 'mine'):
            options = dict(reduction_ratio=reduction_ratio, dilation_value=dilation_value,
                           pool_stride=pool_stride, use_ca=use_ca, use_sa=use_sa)
        else:
            options = dict()

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, **options))
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer, **options))

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


def my_resnet34_cifar100(reduction_ratio=16, dilation_value=2, pool_stride=2, use_ca=True, use_sa=True):
    """   Constructs a ResNet-34 model with 100 classes with my attention module   """
    model = MyResNet(MyBasicBlock, num_layers=[3, 4, 6, 3], num_classes=100, reduction_ratio=reduction_ratio,
                     dilation_value=dilation_value, pool_stride=pool_stride, use_ca=use_ca, use_sa=use_sa)
    return model


def my_resnet50_cifar100(reduction_ratio=16, dilation_value=2, pool_stride=2, use_ca=True, use_sa=True):
    """   Constructs a ResNet-50 model with 100 classes with my attention module   """

    model = MyResNet(MyBottleneck, [3, 4, 6, 3], num_classes=100, reduction_ratio=reduction_ratio,
                     dilation_value=dilation_value, pool_stride=pool_stride, use_ca=use_ca, use_sa=use_sa)
    return model

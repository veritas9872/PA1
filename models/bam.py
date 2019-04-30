import torch.nn as nn
from models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


def dil_conv1x1(in_planes, out_planes, bias=True):
    """
    1x1 convolution. Dilation has no meaning for 1x1 conv.
    Has default True for bias, unlike in resnet and SENet.
    Named it differently because conv1x1 was already taken.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)


def dil_conv3x3(in_planes, out_planes, dilation, groups=1, bias=True):
    """3x3 dilated convolution with padding and fixed stride at 1"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, groups=groups, bias=bias)


class CALayer(nn.Module):
    """
    Channel Attention Layer
    """
    bam = True

    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        num_reduced = num_channels // reduction_ratio
        self.layer = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_reduced),
            nn.ReLU(),
            nn.Linear(in_features=num_reduced, out_features=num_channels),
            nn.BatchNorm1d(num_features=num_channels)
        )

    def forward(self, tensor):
        shape = tensor.shape
        # Global average pooling. Remove dimensions for feeding to linear.
        features = tensor.mean(dim=(-2, -1), keepdim=False)
        channel_attention = self.layer(features).view(shape[0], shape[1], 1, 1).expand_as(tensor)
        # Return scale layer.
        return channel_attention


class SALayer(nn.Module):
    """
    Spatial Attention Layer
    """
    bam = True

    def __init__(self, num_channels, reduction_ratio=16, dilation_value=4):
        super().__init__()

        num_reduced = num_channels // reduction_ratio
        self.layer = nn.Sequential(
            dil_conv1x1(in_planes=num_channels, out_planes=num_reduced, bias=True),
            nn.ReLU(),
            dil_conv3x3(in_planes=num_reduced, out_planes=num_reduced, dilation=dilation_value, bias=True),
            nn.ReLU(),
            dil_conv3x3(in_planes=num_reduced, out_planes=num_reduced, dilation=dilation_value, bias=True),
            nn.ReLU(),
            dil_conv1x1(in_planes=num_reduced, out_planes=1, bias=True),
            nn.BatchNorm2d(num_features=1)
        )

    def forward(self, tensor):
        shape = tensor.shape
        spatial_attention = self.layer(tensor).view(shape[0], 1, shape[2], shape[3]).expand_as(tensor)
        return spatial_attention


class BAMLayer(nn.Module):

    bam = True

    def __init__(self, num_channels, reduction_ratio=16, dilation_value=4, use_ca=True, use_sa=True):
        """
        Bottleneck Attention Module Layer.

        :param reduction_ratio: Number of channels used in the conv layer which is having BAM applied to it.
        :param reduction_ratio (int): Reduction ratio for both channel and spatial reduction.
        Using the same value for simplicity.
        :param dilation_value (int): Dilation ratio for dilated 3x3 convolution for spatial attention.
        :param use_ca: Whether to use channel attention or not. For ablation studies.
        :param use_sa: Whether to use spatial attention or not. For ablation studies.
        """
        super().__init__()
        self.use_ca = use_ca
        self.use_sa = use_sa

        self.channel_attention = CALayer(num_channels, reduction_ratio)
        self.spatial_attention = SALayer(num_channels, reduction_ratio, dilation_value)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):

        if self.use_ca and self.use_sa:
            attention = self.sigmoid(self.channel_attention(tensor) + self.spatial_attention(tensor))

        elif not (self.use_ca or self.use_sa):
            attention = 1

        elif self.use_ca and not self.use_sa:
            attention = self.sigmoid(self.channel_attention(tensor))

        elif not self.use_ca and self.use_sa:
            attention = self.sigmoid(self.spatial_attention(tensor))

        else:
            raise SyntaxError('Impossible combination')

        return tensor * attention


class BAMResNet(nn.Module):

    def __init__(self, blocks, num_layers, num_classes=100, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 reduction_ratio=16, dilation_value=4, use_ca=True, use_sa=True):

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

        if issubclass(blocks, nn.Module):
            assert hasattr(blocks, 'expansion'), 'All blocks must have an "expansion" attribute.'
            blocks = [blocks] * 4

        elif len(blocks) == 4:
            for block in blocks:
                assert issubclass(block, nn.Module), 'Elements must be Pytorch Modules.'
                assert hasattr(block, 'expansion'), 'All blocks must have an "expansion" attribute.'

        else:
            raise NotImplementedError('This should be a list of 4 block types or of one block type to be used.')

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[0], 64, num_layers[0])
        self.bam1 = BAMLayer(64*blocks[0].expansion, reduction_ratio, dilation_value, use_ca, use_sa)
        self.layer2 = self._make_layer(blocks[1], 128, num_layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.bam2 = BAMLayer(128*blocks[1].expansion, reduction_ratio, dilation_value, use_ca, use_sa)
        self.layer3 = self._make_layer(blocks[2], 256, num_layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.bam3 = BAMLayer(256*blocks[2].expansion, reduction_ratio, dilation_value, use_ca, use_sa)
        self.layer4 = self._make_layer(blocks[3], 512, num_layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocks[3].expansion, num_classes)

        if use_ca or use_sa:  # Use BAM if either of the options is on.
            self.use_bam = True
        else:
            self.use_bam = False

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.bam1(x) if self.use_bam else x  # I believe that this is where BAM is supposed to be located.
        x = self.layer2(x)
        x = self.bam2(x) if self.use_bam else x
        x = self.layer3(x)
        x = self.bam3(x) if self.use_bam else x
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def bam_resnet50_cifar100(use_ca=True, use_sa=True):
    blocks = Bottleneck  # Default settings in official implementation.
    model = BAMResNet(blocks=blocks, num_layers=[3, 4, 6, 3], num_classes=100, use_ca=use_ca, use_sa=use_sa)
    return model




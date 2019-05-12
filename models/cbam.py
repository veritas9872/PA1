import torch.nn as nn
from models.resnet import conv1x1, conv3x3, Bottleneck, BasicBlock
import torch


class CALayer(nn.Module):
    """
    Channel Attention Layer for CBAM.
    """
    is_cbam = True

    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling
        self.gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))  # Global Maximum Pooling

        self.layer = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_features=num_channels // reduction_ratio, out_features=num_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        shape = tensor.shape
        # Remove dimensions for feeding to linear.
        gap = self.gap(tensor).view(shape[0], shape[1])
        gmp = self.gmp(tensor).view(shape[0], shape[1])
        features = torch.cat((gap, gmp), dim=0)  # Concat on batch axis for parallel calculation on FC layer.
        # Apparently, torch.chunk cannot be jit compiled.
        af, mf = torch.chunk(self.layer(features), chunks=2, dim=0)  # Split the results.
        channel_attention = self.sigmoid(af + mf).view(shape[0], shape[1], 1, 1).expand_as(tensor)
        return tensor * channel_attention  # af: average features, mf: max features


class SALayer(nn.Module):
    """
    Spatial Attention Layer for CBAM.
    """
    is_cbam = True

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size % 2 == 1
        self.layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, tensor):
        # Keep dimensions for feeding into CNN.
        gap = tensor.mean(dim=1, keepdim=True)  # Average Pooling along channel dimension
        gmp, _ = tensor.max(dim=1, keepdim=True)  # Maximum Pooling along channel dimension
        features = torch.cat((gap, gmp), dim=1)  # Channel-wise concatenation.
        spatial_attention = self.layer(features).expand_as(tensor)
        return tensor * spatial_attention


class CBAMLayer(nn.Module):
    is_cbam = True

    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):
        super().__init__()

        self.channel_attention = CALayer(num_channels, reduction_ratio)
        self.spatial_attention = SALayer(kernel_size)

        self.use_ca = use_ca
        self.use_sa = use_sa

    def forward(self, tensor):
        if self.use_ca:
            tensor = self.channel_attention(tensor)
        if self.use_sa:
            tensor = self.spatial_attention(tensor)
        return tensor


class CBAMBasicBlock(nn.Module):
    is_cbam = True
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):

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
        self.downsample = downsample
        self.stride = stride

        self.use_cbam = (use_ca or use_sa)
        self.cbam = CBAMLayer(planes * self.expansion, reduction_ratio, kernel_size, use_ca, use_sa)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:  # Putting this in the same location as the SE module.
            out = self.cbam(out)  # Different from code, paper does not specify where. However, flop number is a hint.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CBAMBottleneck(nn.Module):
    is_cbam = True
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):

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
        self.downsample = downsample
        self.stride = stride

        self.use_cbam = (use_ca or use_sa)
        self.cbam = CBAMLayer(planes * self.expansion, reduction_ratio, kernel_size, use_ca, use_sa)

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

        if self.use_cbam:  # Putting this in the same location as the SE module.
            out = self.cbam(out)  # Different from code, paper does not specify where. However, flop number is a hint.

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CBAMResNet(nn.Module):
    is_cbam = True

    def __init__(self, blocks, num_layers, num_classes=100, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):

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
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        cbam_kwargs = dict(reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_ca=use_ca, use_sa=use_sa)
        self.layer1 = self._make_layer(blocks[0], 64, num_layers[0], **cbam_kwargs)
        self.layer2 = self._make_layer(blocks[1], 128, num_layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], **cbam_kwargs)
        self.layer3 = self._make_layer(blocks[2], 256, num_layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], **cbam_kwargs)
        self.layer4 = self._make_layer(blocks[3], 512, num_layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], **cbam_kwargs)
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
                if isinstance(m, (Bottleneck, CBAMBottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False,
                    reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):

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
            if hasattr(block, 'is_cbam') and hasattr(block, 'expansion'):  # CBAM block
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                    dilation=self.dilation, norm_layer=norm_layer, reduction_ratio=reduction_ratio,
                                    kernel_size=kernel_size, use_ca=use_ca, use_sa=use_sa))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                    dilation=self.dilation, norm_layer=norm_layer))

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


def cbam_resnet34_cifar100(reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):
    blocks = CBAMBasicBlock
    model = CBAMResNet(blocks=blocks, num_layers=[3, 4, 6, 3], num_classes=100,
                       reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_ca=use_ca, use_sa=use_sa)
    return model


def cbam_resnet50_cifar100(reduction_ratio=16, kernel_size=7, use_ca=True, use_sa=True):
    blocks = CBAMBottleneck
    model = CBAMResNet(blocks=blocks, num_layers=[3, 4, 6, 3], num_classes=100,
                       reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_ca=use_ca, use_sa=use_sa)
    return model

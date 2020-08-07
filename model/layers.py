import collections
from math import ceil, floor, log2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SIMPLE CONV2D BLOCK WITH PADDING SPECIFIED
class Conv2d(nn.Module):
    """
    Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           padding_type='REFLECTION', dilation=1, groups=1, bias=True)
    if padding is not specified explicitly, compute padding = floor(kernel_size/2)
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__()
        p = 0
        conv_block = []
        kernel_size = args[2]
        dilation = kwargs.pop('dilation', 1)
        padding = kwargs.pop('padding', None)
        if padding is None:
            if isinstance(kernel_size, collections.Iterable):
                assert (len(kernel_size) == 2)
            else:
                kernel_size = [kernel_size] * 2

            padding = (floor((kernel_size[0] - 1) / 2),
                       ceil((kernel_size[0] - 1) / 2),
                       floor((kernel_size[1] - 1) / 2),
                       ceil((kernel_size[1] - 1) / 2))

        try:
            if kwargs['padding_type'] == 'REFLECTION':
                conv_block += [
                    nn.ReflectionPad2d(padding),
                ]
            elif kwargs['padding_type'] == 'ZERO':
                p = padding
            elif kwargs['padding_type'] == 'REPLICATE':
                conv_block += [
                    nn.ReplicationPad2d(padding),
                ]

        except KeyError as e:
            # use default padding 'REFLECT'
            conv_block += [
                nn.ReflectionPad2d(padding),
            ]
        except Exception as e:
            raise e

        conv_block += [
            nn.Conv2d(*args, padding=p, dilation=dilation, **kwargs)
        ]
        self.conv = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv(x)


class PixelShuffleUpsampler(nn.Sequential):
    """Upsample block with pixel shuffle"""

    # ratio must be multiple of 2
    def __init__(self, ratio, total_resnet_features):
        super(PixelShuffleUpsampler, self).__init__()
        layers = []
        for i in range(int(log2(ratio))):
            # 2x upsampling
            layers += [Conv2d(total_resnet_features, 4 * total_resnet_features, 3), nn.PixelShuffle(2)]
            layers.append(nn.ReLU(inplace=True))

        self.m = nn.Sequential(*layers)

# A CONV(1,1) BLOCK
class CompressionBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(CompressionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = super(CompressionBlock, self).forward(x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class DenseLayer(nn.Sequential):
    # bn_size 4
    # num_input_features = 160
    # growth_rate=40
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        num_output_features = bn_size * growth_rate

        self.add_module(
            'conv_1',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=True)),

        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv_2',
            Conv2d(num_output_features, growth_rate, 3, stride=1, bias=True)),

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        # RESIDUAL CONNECTION
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    # res_factor,num_layers,num_input_features,bn_size,growth_rate
    # num_layers IS 8 -THE NUMBER OF CONSECUTIVE DENSE LAYERS
    # bn_size 4
    # num_input_features = num_init_features = 160
    # growth_rate=40
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)


# DENSELY_CONNECTED_BLOCKS
class DenseConnectedBlock(nn.Sequential):
    # res_factor,num_layers,num_input_features,bn_size,growth_rate
    def __init__(self, **kwargs):
        super(DenseConnectedBlock, self).__init__()
        self.res_factor = kwargs.pop('res_factor')

        # THESE ARE THE 8 DENSE_LAYERS
        self.dense_block = _DenseBlock(**kwargs)

        #160+8*40
        num_features = kwargs['num_input_features'] + kwargs['num_layers'] * kwargs['growth_rate']

        # A COMPRESSION BLOCK AT THE END OF EACH DCU - KEEPS OUTPUT SIZE TO 160
        # THIS IS THE CONV(1,1) AT THE END OF DCU
        self.comp = CompressionBlock(
            in_planes=num_features,
            out_planes=kwargs['num_input_features'],
        )

    def forward(self, x, identity_x=None):
        if identity_x is None:
            identity_x = x
        return self.res_factor * super(DenseConnectedBlock, self).forward(x) + identity_x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        # m.weight.data.normal_(0, sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        b, c, h, w = m.weight.data.size()
        f = ceil(w / 2)
        cen = (2 * f - 1 - f % 2) / (2.0 * f)
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        fil = (1 - np.abs(xv / f - cen)) * (1 - np.abs(yv / f - cen))
        fil = fil[np.newaxis, np.newaxis, ...]
        fil = np.repeat(fil, 3, 0)
        m.weight.data.copy_(torch.from_numpy(fil))



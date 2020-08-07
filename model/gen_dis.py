import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import vgg19
from math import log2
from model.layers import PixelShuffleUpsampler,Conv2d,CompressionBlock,init_weights,DenseConnectedBlock
from collections import OrderedDict

class HierarchicalGenerator(nn.Module):

    def __init__(self, num_init_features, bn_size, growth_rate, ps_woReLU, level_config,
                 res_factor, max_num_feature, max_scale, **kwargs):
        super(HierarchicalGenerator,self).__init__()
        self.max_upscale = max_scale
        self.num_pyramids = int(log2(self.max_upscale))

        self.current_scale_idx = self.num_pyramids - 1

        self.Upsampler = PixelShuffleUpsampler
        self.upsample_args = {'woReLU': ps_woReLU}

        denseblock_params = {
            'num_layers': None,
            'num_input_features': num_init_features,
            'bn_size': bn_size,
            'growth_rate': growth_rate,
        }

        num_features = denseblock_params['num_input_features']  # 160

        # Initiate network

        # each scale has its own init_conv - V ******************************************************
        for s in range(1, self.num_pyramids + 1):
            # in_channel=3, out_channel=160
            self.add_module('init_conv_%d' % s, Conv2d(3, num_init_features, 3))

        # Each denseblock forms a pyramid - 0,1,2
        for i in range(self.num_pyramids):
            block_config = level_config[i]  # LIST LIKE [8,8,8,...,8]
            pyramid_residual = OrderedDict()

            # AT THE END OF EACH DCU, WE INCLUDE A CONV(1,1) COMPRESSION LAYER
            # NO NEED FOR THIS AT LEVEL 0
            if i != 0:
                out_planes = num_init_features

                # out_planes = num_init_features = 160
                pyramid_residual['compression_%d' % i] = CompressionBlock(in_planes=num_features, out_planes=out_planes)
                num_features = out_planes

            # serial connect blocks
            # NUM OF ELEMENTS IN block_confis IS THE NUMBER OF DCUs IN THAT PYRAMID
            # CREATING 8 DCUs
            for b, num_layers in enumerate(block_config):
                # FOR EACH DENSELY_CONNECTED_UNIT ***********************************************************
                # num_layers IS ALWAYS 8
                denseblock_params['num_layers'] = num_layers
                denseblock_params['num_input_features'] = num_features  # 160

                # DENSELY_CONNECTED_BLOCK WITH CONV(1,1) INSIDE*********************************************************
                pyramid_residual['residual_denseblock_%d' %(b + 1)] = DenseConnectedBlock(
                    res_factor=res_factor,
                    **denseblock_params)

            # conv before upsampling
            # THIS IS R
            block, num_features = self.create_finalconv(num_features, max_num_feature)

            # CREATING PYRAMID
            pyramid_residual['final_conv'] = block
            self.add_module('pyramid_residual_%d' % (i + 1),
                            nn.Sequential(pyramid_residual))

            # upsample the residual by 2 before reconstruction and next level
            self.add_module(
                'pyramid_residual_%d_residual_upsampler' % (i + 1),
                self.Upsampler(2, num_features))

            # reconstruction convolutions
            reconst_branch = OrderedDict()
            out_channels = num_features
            reconst_branch['final_conv'] = Conv2d(out_channels, 3, 3)
            self.add_module('reconst_%d' % (i + 1),
                            nn.Sequential(reconst_branch))

        init_weights(self)

    # GET V BASED ON PYRAMID-ID
    def get_init_conv(self, idx):
        """choose which init_conv based on curr_scale_idx (1-based)"""
        return getattr(self, 'init_conv_%d' % idx)

    def create_finalconv(self, in_channels, max_channels=None):
        block = OrderedDict()
        if in_channels > max_channels:
            # OUR PATH
            block['final_comp'] = CompressionBlock(in_channels, max_channels)
            block['final_conv'] = Conv2d(max_channels, max_channels, (3, 3))
            out_channels = max_channels
        else:
            block['final_conv'] = Conv2d(in_channels, in_channels, (3, 3))
            out_channels = in_channels
        return nn.Sequential(block), out_channels

    # UPSCALE_FACTOR IS THE CURRENT MODEL SCALE...FROM THE DATASET
    def forward(self, x, upscale_factor=None, blend=1.0):
        if upscale_factor is None:
            upscale_factor = self.max_scale
        else:
            valid_upscale_factors = [
                2 ** (i + 1) for i in range(self.num_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                print("Invalid upscaling factor {}: choose one of: {}".format(upscale_factor, valid_upscale_factors))
                raise SystemExit(1)

        #print("RIKI: gen_dis UPSCALE FACTOR: " + str(upscale_factor))
        # GET THE V FOR THIS UPSCALE FACTOR
        # V- COMPUTATION **********************
        feats = self.get_init_conv(log2(upscale_factor))(x)

        # THIS ENSURES WE ONLY GO DOWN THE RELEVANT PART OF THE PYRAMID
        #print(">>>>>UPSCALE:" +str(upscale_factor))
        for s in range(1, int(log2(upscale_factor)) + 1):

            # PYRAMID- COMPUTATION **********************
            feats = getattr(self, 'pyramid_residual_%d' % s)(feats) + feats

            # UPSAMPLING- COMPUTATION **********************
            feats = getattr(self, 'pyramid_residual_%d_residual_upsampler' % s)(feats)

            # RECONSTRUCTION **********************
            # reconst residual image if reached desired scale /
            # use intermediate as base_img / use blend and s is one step lower than desired scale
            if 2 ** s == upscale_factor or (blend != 1.0 and 2 ** (s + 1) == upscale_factor):
                tmp = getattr(self, 'reconst_%d' % s)(feats)
                # if using blend, upsample the second last feature via bilinear upsampling
                if (blend != 1.0 and s == self.current_scale_idx):
                    #print("HERE: SCALEID:"+str(self.current_scale_idx))
                    base_img = nn.functional.upsample(
                        tmp,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)
                if 2 ** s == upscale_factor:
                    if (blend != 1.0) and s == self.current_scale_idx + 1:
                        tmp = tmp * blend + (1 - blend) * base_img
                    output = tmp

        return output


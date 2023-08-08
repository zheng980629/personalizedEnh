import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, upsample_and_concat
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class decomp_KinD(nn.Module):
    """Decomposition network structure(KinD).

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_feat=64,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(decomp_KinD, self).__init__()

        self.conv1 = ConvLReLUNoBN(num_in_ch, num_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = ConvLReLUNoBN(num_feat, num_feat * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv3 = ConvLReLUNoBN(num_feat * 2, num_feat * 4)

        self.up1 = upsample_and_concat(num_feat * 4, num_feat * 2)
        self.decoder_conv1 = ConvLReLUNoBN(num_feat * 4, num_feat * 2)

        self.up2 = upsample_and_concat(num_feat * 2, num_feat)
        self.decoder_conv2 = ConvLReLUNoBN(num_feat * 2, num_feat)

        self.Reflect_out = ConvLReLUNoBN(num_feat, 3, kernel=1, padding=0, act=False)

        self.lconv2 = ConvLReLUNoBN(num_feat, num_feat)
        self.lconv3 = ConvLReLUNoBN(num_feat * 2, num_feat)
        self.Illu_out = ConvLReLUNoBN(num_feat, 1, kernel=1, padding=0, act=False)

    def forward(self, x):

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)

        up8 = self.up1(conv3, conv2)
        conv8 = self.decoder_conv1(up8)

        up9 = self.up2(conv8, conv1)
        conv9 = self.decoder_conv2(up9)

        conv10 = self.Reflect_out(conv9)
        Re_out = torch.sigmoid(conv10)

        l_conv2 = self.lconv2(conv1)
        l_conv3 = torch.cat((F.upsample_nearest(l_conv2, conv9.shape[-2:]), conv9), dim=1)
        l_conv4 = self.lconv3(l_conv3)
        l_conv5 = self.Illu_out(l_conv4)
        Illu_out = torch.sigmoid(l_conv5)

        return Re_out, Illu_out
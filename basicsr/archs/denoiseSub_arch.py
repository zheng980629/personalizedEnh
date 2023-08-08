import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, upsample_and_concat, \
Noise2Noise_ConvBlock, Half_Exposure_Interactive_Modulation, Interactive_Modulation
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Noise2NoiseSubtraction(nn.Module):
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
    def __init__(self, num_in_ch, output_channels=64, conditional_dim=16, finetune=False):
        super(Noise2NoiseSubtraction, self).__init__()

        self.conv1 = Noise2Noise_ConvBlock(num_in_ch, output_channels, 3)
        self.conv1_2 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)
        # self.BGSFM1 = BrightnessGuided_Spatial_Feature_Modulation(output_channels)
        self.pool1 = Noise2Noise_ConvBlock(output_channels, output_channels, ks=3, stride=2)

        self.conv2 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)
        # self.BGSFM2 = BrightnessGuided_Spatial_Feature_Modulation(output_channels)
        self.pool2 = Noise2Noise_ConvBlock(output_channels, output_channels, ks=3, stride=2)

        self.conv2_2 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)
        # self.BGSFM3 = BrightnessGuided_Spatial_Feature_Modulation(output_channels)

        self.conv3 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)
        self.conv3_1 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)

        self.deConv1 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deConv1_2 = Noise2Noise_ConvBlock(output_channels * 2, output_channels, 3)
        self.deConv1_3 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)
        # self.BGSFM4 = BrightnessGuided_Spatial_Feature_Modulation(output_channels * 2)
        # self.BGSFM4_downChannel = Noise2Noise_ConvBlock(output_channels * 2, output_channels, 1)
        self.BGSFM4_downChannel = Noise2Noise_ConvBlock(output_channels, output_channels, 3)

        self.deConv2 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deConv2_2 = Noise2Noise_ConvBlock(output_channels * 2, output_channels, 3)
        self.deConv2_3 = Noise2Noise_ConvBlock(output_channels, output_channels, 3)
        # self.BGSFM5 = BrightnessGuided_Spatial_Feature_Modulation(output_channels * 2)
        # self.BGSFM5_downChannel = Noise2Noise_ConvBlock(output_channels * 2, output_channels, 1)
        self.BGSFM5_downChannel = Noise2Noise_ConvBlock(output_channels, output_channels, 3)

        self.outputConv = nn.Conv2d(output_channels, 3, 3, stride=1, padding=1)
        self.upsample = F.interpolate

        self.condition_net = nn.Sequential(
            nn.Linear(1, conditional_dim, bias=False)
        )

        if finetune:
            for p in self.parameters():
                p.requires_grad = False

        self.BGSFM1 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM1_2 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM2 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM3 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM3_2 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM4 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM4_2 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM5 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)
        self.BGSFM5_2 = Interactive_Modulation(vector_dim=conditional_dim, feature_channel=output_channels)

    def forward(self, input, control):

        modulation_vector = self.condition_net(control)
        input_max_brightness = torch.max(input, dim=1)[0].unsqueeze(dim=1)
        input = torch.cat((input, input_max_brightness), dim=1)

        conv1 = self.conv1(input)
        conv1_modulated = self.BGSFM1(conv1, modulation_vector) * conv1
        conv1_2 = self.conv1_2(conv1_modulated)
        conv1_modulated = self.BGSFM1(conv1_2, modulation_vector) * conv1_2
        conv1_2_res_down = self.pool1(conv1_modulated)

        conv2 = self.conv2(conv1_2_res_down)
        conv2_modulated = self.BGSFM2(conv2, modulation_vector) * conv2
        conv2_res_down = self.pool2(conv2_modulated)

        conv3 = self.conv3(conv2_res_down)
        conv3_modulated = self.BGSFM3(conv3, modulation_vector) * conv3
        conv3_1 = self.conv3_1(conv3_modulated)
        conv3_1_modulated = self.BGSFM3_2(conv3_1, modulation_vector) * conv3_1

        deConv1 = self.deConv1(conv3_1_modulated)
        deConv1 = self.upsample(deConv1, size=conv2_modulated.shape[-2:], mode='bilinear')
        deConv1 = self.deConv1_2(torch.cat((deConv1, conv2_modulated), dim=1))
        deConv1_modulated = self.BGSFM4(deConv1, modulation_vector) * deConv1
        deConv1_2 = self.deConv1_3(deConv1_modulated)
        deConv1_2_modulated = self.BGSFM4_downChannel(self.BGSFM4_2(deConv1_2, modulation_vector) * deConv1_2)

        deConv2 = self.deConv2(deConv1_2_modulated)
        deConv2 = self.upsample(deConv2, size=conv1_modulated.shape[-2:], mode='bilinear')
        deConv2 = self.deConv2_2(torch.cat((deConv2, conv1_modulated), dim=1))
        deConv2_modulated = self.BGSFM5(deConv2, modulation_vector) * deConv2
        deConv2_2 = self.deConv2_3(deConv2_modulated)
        deConv2_modulated = self.BGSFM5_downChannel(self.BGSFM5_2(deConv2_2, modulation_vector) * deConv2_2)

        output = self.outputConv(deConv2_modulated)

        return output


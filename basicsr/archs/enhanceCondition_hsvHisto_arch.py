import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, \
upsample_and_concat, Half_Illumination_Interactive_Modulation, simple_batch_norm_1d, Conv3x3Stack, DConv3x3Stack
from basicsr.utils.registry import ARCH_REGISTRY

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class HSVSimilarityCondition(nn.Module):
    def __init__(self, in_nc=2, nf=24):
        super(HSVSimilarityCondition, self).__init__()
        stride = 2
        pad = 0
        self.cond = nn.Sequential(
            nn.Linear(in_nc, nf // 6, bias=False), 
            nn.Linear(nf // 6, nf // 4, bias=False), 
            nn.Linear(nf // 4, nf // 2, bias=False), 
            nn.Linear(nf // 2, nf, bias=False)
        )

    def forward(self, x):

        return self.cond(x)

@ARCH_REGISTRY.register()
class EnhancementCondition_hsvHistogram(nn.Module):
    """enhancement network structure, processing the illumination map and the reflection map.

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
                 illu_num_in_ch,
                 illu_num_feat=16,
                 illu_histogram_bins=256, 
                 illu_histogram_dim=64,
                 illu_num_out_ch=1,
                 condition_num_in_ch=3, 
                 condition_hidden_ch=64,
                 negative_slope=0.2,
                 reflection_num_in_ch=3, 
                 reflection_num_base=16, 
                 reflection_num_out_ch=3, 
                 tanh=False, 
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EnhancementCondition_hsvHistogram, self).__init__()

        self.reflection_num_base = reflection_num_base


        ################### illumiantion mapping ###################
        self.illu_conv1 = ConvLReLUNoBN(illu_num_in_ch, illu_num_feat)
        self.illu_IIM1 = Half_Illumination_Interactive_Modulation(illu_num_feat, illu_histogram_dim)

        self.illu_conv2 = ConvLReLUNoBN(illu_num_feat, illu_num_feat*2)
        self.illu_IIM2 = Half_Illumination_Interactive_Modulation(illu_num_feat*2, illu_histogram_dim)

        self.illu_conv3 = ConvLReLUNoBN(illu_num_feat*2, illu_num_feat*2)
        self.illu_IIM3 = Half_Illumination_Interactive_Modulation(illu_num_feat*2, illu_histogram_dim)

        self.illu_conv4 = ConvLReLUNoBN(illu_num_feat*2, illu_num_feat)
        self.illu_IIM4 = Half_Illumination_Interactive_Modulation(illu_num_feat, illu_histogram_dim)

        self.illu_conv5 = ConvLReLUNoBN(illu_num_feat, illu_num_out_ch)

        self.illu_histogram_average_condition = nn.Linear(illu_histogram_bins, illu_histogram_dim, bias=False)
        self.bn = nn.InstanceNorm1d(num_features=illu_histogram_bins, affine=False)

        ################### condition network ###################
        self.cond_hsvSimilarity = HSVSimilarityCondition(in_nc=2, nf=reflection_num_base * 4)

        self.cond_scale1 = nn.Linear(reflection_num_base * 4, reflection_num_base * 8, bias=True)
        self.cond_scale2 = nn.Linear(reflection_num_base * 4, reflection_num_base * 4,  bias=True)
        self.cond_scale3 = nn.Linear(reflection_num_base * 4, reflection_num_base * 2, bias=True)
        self.cond_scale4 = nn.Linear(reflection_num_base * 4, reflection_num_base, bias=True)

        self.cond_shift1 = nn.Linear(reflection_num_base * 4, reflection_num_base * 8, bias=True)
        self.cond_shift2 = nn.Linear(reflection_num_base * 4, reflection_num_base * 4, bias=True)
        self.cond_shift3 = nn.Linear(reflection_num_base * 4, reflection_num_base * 2, bias=True)
        self.cond_shift4 = nn.Linear(reflection_num_base * 4, reflection_num_base, bias=True)

        ################### reflection mapping ###################
        filters = [reflection_num_base, reflection_num_base * 2, reflection_num_base * 4, reflection_num_base * 8, reflection_num_base * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(reflection_num_in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], reflection_num_out_ch, kernel_size=1, stride=1, padding=0)

        self.tanh = tanh

    def forward(self, lq_illu, ref_illu, ref_histogram, lq_refl, ref_refl, hue_similarity, saturation_similarity):
        
        ################### illumiantion mapping ###################
        ref_histogram = 1000000 * ref_histogram
        ref_histogram = simple_batch_norm_1d(ref_histogram) * 0.1
        histogram_vector = self.illu_histogram_average_condition(ref_histogram)

        illu_enhanced_conv1 = self.illu_conv1(lq_illu)
        illu_enhanced_conv1_modu = self.illu_IIM1(illu_enhanced_conv1, histogram_vector)

        illu_enhanced_conv2 = self.illu_conv2(illu_enhanced_conv1_modu)
        illu_enhanced_conv2_modu = self.illu_IIM2(illu_enhanced_conv2, histogram_vector)

        illu_enhanced_conv3 = self.illu_conv3(illu_enhanced_conv2_modu)
        illu_enhanced_conv3_modu = self.illu_IIM3(illu_enhanced_conv3, histogram_vector)

        illu_enhanced_conv4 = self.illu_conv4(illu_enhanced_conv3_modu)
        illu_enhanced_conv4_modu = self.illu_IIM4(illu_enhanced_conv4, histogram_vector)

        illu_enhanced_out = self.illu_conv5(illu_enhanced_conv4_modu)
        illu_enhanced_out = torch.sigmoid(illu_enhanced_out)

        ################### reflection mapping ###################

        ############ learnable adaptive instance ############ 
        # cond = self.Conv5(self.Maxpool4(self.Conv4(self.Maxpool3(self.Conv3(self.Maxpool2(self.Conv2(self.Maxpool1(self.Conv1(ref_refl)))))))))
        # cond = torch.mean(cond, dim=[2, 3], keepdim=False)
        # hs_similarity = torch.cat((hue_similarity.unsqueeze(0).unsqueeze(0), saturation_similarity.unsqueeze(0).unsqueeze(0)), dim=1)
        hs_similarity = torch.cat((hue_similarity, saturation_similarity), dim=1)
        cond = self.cond_hsvSimilarity(hs_similarity)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)
        
        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)
        
        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)
        
        scale4 = self.cond_scale4(cond)
        shift4 = self.cond_shift4(cond)

        ############ low-light ############
        # encoder
        e1 = self.Conv1(lq_refl)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # e5 = self.adaptive_instance_normalization(e5, cond)

        # decoder
        d5 = self.Up5(e5)
        d5 = F.interpolate(d5, size=e4.shape[-2:], mode='bilinear')
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = d5 * scale1.view(-1, self.reflection_num_base * 8, 1, 1) + shift1.view(-1, self.reflection_num_base * 8, 1, 1) + d5


        d4 = self.Up4(d5)
        d4 = F.interpolate(d4, size=e3.shape[-2:], mode='bilinear')
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = d4 * scale2.view(-1, self.reflection_num_base * 4, 1, 1) + shift2.view(-1, self.reflection_num_base * 4, 1, 1) + d4


        d3 = self.Up3(d4)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear')
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = d3 * scale3.view(-1, self.reflection_num_base * 2, 1, 1) + shift3.view(-1, self.reflection_num_base * 2, 1, 1) + d3


        d2 = self.Up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear')
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = d2 * scale4.view(-1, self.reflection_num_base, 1, 1) + shift4.view(-1, self.reflection_num_base, 1, 1) + d2
        
        lq_reflection_out = torch.sigmoid(self.Conv(d2))

        return illu_enhanced_out, lq_reflection_out, cond


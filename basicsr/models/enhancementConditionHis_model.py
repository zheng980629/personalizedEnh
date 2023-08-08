import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.loss_util import histcal, histcal_tensor, standardization, noiseMap, rgb2lab, rgb2hsv
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, noise_estimate_batch
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import sys
np.set_printoptions(threshold=np.inf)
@MODEL_REGISTRY.register()
class EnhanceConditionHisModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(EnhanceConditionHisModel, self).__init__(opt)

        # define network
        self.net_decom = build_network(opt['network_decom'])
        self.net_decom = self.model_to_device(self.net_decom)
        self.print_network(self.net_decom)

        self.net_denoise = build_network(opt['network_denoise'])
        self.net_denoise = self.model_to_device(self.net_denoise)
        self.print_network(self.net_denoise)

        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path_decom = self.opt['path'].get('pretrain_network_decom', None)
        if load_path_decom is not None:
            param_key = self.opt['path'].get('param_key_decom', 'params')
            self.load_network(self.net_decom, load_path_decom, self.opt['path'].get('strict_load_decom', True), param_key)

        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        if load_path_g is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path_g, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_denoise = self.opt['path'].get('pretrain_network_denoise', None)
        if load_path_denoise is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_denoise, load_path_denoise, self.opt['path'].get('strict_load_denoise', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        ######################### illumination #########################
        if train_opt['pixel_opt']['loss_weight'] > 0:
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt['histogram_opt']['loss_weight'] > 0:
            self.cri_hist = build_loss(train_opt['histogram_opt']).to(self.device)
        else:
            logger.info('Remove histogram loss.')
            self.cri_hist = None

        if train_opt['spatial_opt']['loss_weight'] > 0:
            self.cri_spa = build_loss(train_opt['spatial_opt']).to(self.device)
        else:
            self.cri_spa = None
            logger.info('Remove spa loss.')

        if train_opt['color_opt']['loss_weight'] > 0:
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None
            logger.info('Remove color loss.')

        if train_opt['Lab_opt']['loss_weight'] > 0:
            self.cri_Lab = build_loss(train_opt['Lab_opt']).to(self.device)
        else:
            self.cri_Lab = None
            logger.info('Remove Lab color space loss.')


        ######################### reflection #########################
        if train_opt['colorMapHis_opt']['loss_weight'] > 0:
            self.cri_colorMapHist = build_loss(train_opt['colorMapHis_opt']).to(self.device)
        else:
            logger.info('Remove reflection color map histogram loss.')
            self.cri_colorMapHist = None

        if train_opt['hsvReflHis_opt']['loss_weight'] > 0:
            self.cri_hsvReflHis = build_loss(train_opt['hsvReflHis_opt']).to(self.device)
        else:
            logger.info('Remove reflection histogram loss in the HSV space.')
            self.cri_hsvReflHis = None

        if train_opt['meanReflHis_opt']['loss_weight'] > 0:
            self.cri_meanReflHist = build_loss(train_opt['meanReflHis_opt']).to(self.device)
        else:
            logger.info('Remove mean reflection map histogram loss.')
            self.cri_meanReflHist = None

        if train_opt['colorMapGram_opt']['loss_weight'] > 0:
            self.cri_colorMapGram = build_loss(train_opt['colorMapGram_opt']).to(self.device)
        else:
            logger.info('Remove  reflection color map gram matrics loss.')
            self.cri_colorMapGram = None

        if train_opt['reflGram_opt']['loss_weight'] > 0:
            self.cri_reflectionGram = build_loss(train_opt['reflGram_opt']).to(self.device)
        else:
            logger.info('Remove reflection map gram matrics loss.')
            self.cri_reflectionGram = None

        if train_opt['spatialRefl_opt']['loss_weight'] > 0:
            self.cri_spaRefl = build_loss(train_opt['spatialRefl_opt']).to(self.device)
        else:
            self.cri_spaRefl = None
            logger.info('Remove spa reflection loss.')

        if train_opt['colorRefl_opt']['loss_weight'] > 0:
            self.cri_colorRefl = build_loss(train_opt['colorRefl_opt']).to(self.device)
        else:
            self.cri_colorRefl = None
            logger.info('Remove reflection color loss.')

        if train_opt['perceptual_opt']['perceptual_weight'] > 0:
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
            logger.info('Remove perceptual loss.')

        if train_opt['perceptualLuminance_opt']['perceptual_weight'] > 0:
            self.cri_perceptualLuminance = build_loss(train_opt['perceptualLuminance_opt']).to(self.device)
        else:
            self.cri_perceptualLuminance = None
            logger.info('Remove perceptual luminance loss.')

        if train_opt['refReflIdentity_opt']['loss_weight'] > 0:
            self.cri_referenceReflIdentity = build_loss(train_opt['refReflIdentity_opt']).to(self.device)
        else:
            self.cri_referenceReflIdentity = None
            logger.info('Remove the reflection of the reference image identity loss.')

        if train_opt['gan_opt']['loss_weight'] > 0:
            self.criterionGAN = build_loss(train_opt['gan_opt']).to(self.device)
            # G_update_ratio and G_init_iters
            self.G_update_ratio = train_opt['G_update_ratio'] if train_opt['G_update_ratio'] else 1
            self.G_init_iters = train_opt['G_init_iters'] if train_opt['G_init_iters'] else 0
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
        else:
            self.criterionGAN = None
            logger.info('Remove gan loss.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_g_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_g_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def augmentation(self, input): # randomly augment some input
        aug_seed = torch.rand(1)
        if aug_seed<0.001:
            # adaptive Gaussian Noise
            bg_noise_std = 0.3 * (0.2+0.8*torch.rand(1).item()) * torch.std(input, dim=[1,2,3], keepdim=True)
            ada_noise_std = 0.04 * input.clamp(max=0.5)
            input_aug = (input + bg_noise_std*torch.randn_like(input) + ada_noise_std*torch.randn_like(input)).clamp_(min=0., max=1.)
        elif aug_seed < 0.001:
            # quantization error
            stairs = 64
            input_aug = torch.floor(input*stairs)/stairs
        else:
            input_aug = input

        return input_aug

    def feed_data(self, data, GT=True, ref=True):
        # self.lq = data['lq'].to(self.device)
        # if 'gt' in data:
        #     self.gt = data['gt'].to(self.device)

        self.real_H, self.ref, self.ref_alt = None, None, None

        self.lq = data['lq'].to(self.device)  # LQ
        if ref and 'ref' in  data:
            self.ref = data['ref'].to(self.device) # ref
            self.ref_aug = self.augmentation(self.ref)
            self.ref_path = data['ref_path']
        if ref and 'ref_alt' in  data:
            self.ref_alt = data['ref_alt'].to(self.device) # ref
            self.ref_path_alt = data['ref_path_alt']
        if GT:
            self.gt = data['gt'].to(self.device)  # GT

    def backward_G(self, step):
        #
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss between input out
                ###l_g_pix_ref_cycle = self.cri_pix(self.rec_ref_cycle, self.ref)
                l_g_pix_ref = self.cri_pix(self.enhanced_AugRef, self.ref_aug)
                l_g_pix = l_g_pix_ref
                ###l_g_pix = self.l_pix_w * 0.5*(l_g_pix_ref + l_g_pix_ref_cycle)
                l_g_total += l_g_pix
                self.log_dict['l_g_pix'] = l_g_pix.item()
                ## self.log_dict['l_g_pix_fake_ref'] = l_g_pix_fake_ref.item()

            ######################### reflection #########################
            if self.cri_colorMapHist:
                self.enhanced_lqRef_refl_colorMapHis = histcal(self.enhanced_lqRef_refl_colorMap)
                self.decom_ref_ref_colorMapHis = histcal(self.decom_ref_ref_colorMap)
                l_g_colorMapHist = self.cri_colorMapHist(self.enhanced_lqRef_refl_colorMapHis, self.decom_ref_ref_colorMapHis)
                l_g_total += l_g_colorMapHist
                self.log_dict['l_g_colorMapHist'] = l_g_colorMapHist.item()

            if self.cri_hsvReflHis:
                self.enhanced_lqRef_refl_hue = rgb2hsv(self.enhanced_lqRef_refl)[:, 0, :, :].unsqueeze(1)
                self.enhanced_lqRef_refl_saturation = rgb2hsv(self.enhanced_lqRef_refl)[:, 1, :, :].unsqueeze(1)
                self.enhanced_lqRef_refl_hueHisto = histcal_tensor(self.enhanced_lqRef_refl_hue)
                self.enhanced_lqRef_refl_saturationHisto = histcal_tensor(self.enhanced_lqRef_refl_saturation)

                l_g_hsvReflHist = self.cri_hsvReflHis(self.enhanced_lqRef_refl_hueHisto, self.enhanced_lqRef_refl_saturationHisto,
                                                        self.decom_ref_ref_hueHisto, self.decom_ref_ref_saturationHisto,
                                                        self.cos_similarity_hue, self.cos_similarity_saturation)
                l_g_total += l_g_hsvReflHist
                self.log_dict['l_g_hsvReflHist'] = l_g_hsvReflHist.item()

            if self.cri_colorMapGram:
                l_g_colorMapGram = self.cri_colorMapGram(self.enhanced_lqRef_refl_colorMap, self.decom_ref_ref_colorMap)
                l_g_total += l_g_colorMapGram
                self.log_dict['l_g_colorMapGram'] = l_g_colorMapGram.item()

            if self.cri_meanReflHist:
                l_g_meanReflHist = self.cri_meanReflHist(self.enhanced_lqRef_refl.mean(dim=1), self.decom_ref_ref.mean(dim=1))
                l_g_total += l_g_meanReflHist
                self.log_dict['l_g_meanReflHist'] = l_g_meanReflHist.item()

            if self.cri_reflectionGram:
                l_g_reflectionGram = self.cri_reflectionGram(self.enhanced_lqRef_refl, self.decom_ref_ref)
                l_g_total += l_g_reflectionGram
                self.log_dict['l_g_reflectionGram'] = l_g_reflectionGram.item()

            if self.cri_colorRefl:
                l_g_colorRefl = self.cri_colorRefl(self.enhanced_lqRef_refl, self.decom_lq_ref)
                l_g_total += l_g_colorRefl
                self.log_dict['l_g_colorRefl'] = l_g_colorRefl.item()

            if self.cri_spaRefl:
                l_spaRefl = torch.mean(self.cri_spaRefl(self.enhanced_lqRef_refl, self.decom_lq_ref))
                l_g_total += l_spaRefl
                self.log_dict['l_spaRefl'] = l_spaRefl.item()

            if self.cri_perceptual:
                l_perceptual, l_style = self.cri_perceptual(self.enhanced_lqRef_refl, self.decom_lq_ref)
                l_g_total += l_perceptual
                self.log_dict['l_perceptual'] = l_perceptual.item()

            if self.cri_perceptualLuminance:
                self.enhanced_lqRef_refl_luminance = rgb2lab(self.enhanced_lqRef_refl)[:, 0, :, :].repeat(1, 3, 1, 1)
                self.decom_lq_ref_luminance = rgb2lab(self.decom_lq_ref)[:, 0, :, :].repeat(1, 3, 1, 1)
                l_perceptualLuminance, _ = self.cri_perceptualLuminance(self.enhanced_lqRef_refl_luminance, self.decom_lq_ref_luminance)
                l_g_total += l_perceptualLuminance
                self.log_dict['l_perceptualLuminance'] = l_perceptualLuminance.item()

            if self.cri_referenceReflIdentity:
                l_referenceReflIdentity = self.cri_referenceReflIdentity(self.enhanced_ref_refl, self.decom_ref_ref)
                l_g_total += l_referenceReflIdentity
                self.log_dict['l_referenceReflIdentity'] = l_referenceReflIdentity.item()


            ######################### illumination #########################
            if self.cri_hist:
                l_g_hist = self.cri_hist(self.enhanced_lqRef_illu_histogram, self.ref_histogram)
                l_g_total += l_g_hist
                self.log_dict['l_g_hist'] = l_g_hist.item()

            if self.cri_color:
                l_g_color = self.cri_color(self.enhanced_lqRef, self.lq)
                l_g_total += l_g_color
                self.log_dict['l_g_color'] = l_g_color.item()

            if self.cri_spa:
                l_spa = torch.mean(self.cri_spa(self.enhanced_lqRef, self.lq))
                l_g_total += l_spa
                self.log_dict['l_spa'] = l_spa.item()

            if self.cri_Lab:
                l_Lab = self.cri_Lab(self.enhanced_lqRef, self.lq)
                l_g_total += l_Lab
                self.log_dict['l_Lab'] = l_spa.item()

            self.l_g_total = l_g_total
            self.log_dict['l_g_total'] = l_g_total.item()
            return l_g_total


    def backward_D_basic(self, netD, real, fake, ext=''):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake)
        if self.opt['gan_type'] == 'wgan':
            loss_D_real = torch.sigmoid(pred_real).mean()
            loss_D_fake = torch.sigmoid(pred_fake).mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD,
                                                real.data, fake.data)
            D_real = loss_D_real
            D_fake = loss_D_fake
        elif self.opt['gan_type'] == 'ragan':
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), 1.) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), 0.)) / 2
            D_real = torch.mean(torch.sigmoid(pred_real - torch.mean(pred_fake)))
            D_fake = torch.mean(torch.sigmoid(pred_fake - torch.mean(pred_real)))
        else:
            loss_D_real = self.criterionGAN(pred_real, 1.)
            loss_D_fake = self.criterionGAN(pred_fake, 0.)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            D_real = torch.mean(torch.sigmoid(pred_real))
            D_fake = torch.mean(torch.sigmoid(pred_fake))
        self.log_dict['D_real'+ext] = D_real.item()
        self.log_dict['D_fake'+ext] = D_fake.item()
        return loss_D


    def forward(self, current_iter):
        self.adaptivePool = nn.AdaptiveAvgPool2d((self.opt['noiseMap_block'], self.opt['noiseMap_block']))

        self.decom_lq = self.net_decom(self.lq)
        self.decom_lq_ref = self.decom_lq[0]
        self.decom_lq_illu = self.decom_lq[1]

        self.decom_ref = self.net_decom(self.ref)
        self.decom_ref_ref = self.decom_ref[0]
        self.decom_ref_illu = self.decom_ref[1]

        self.decom_refAug = self.net_decom(self.ref_aug)
        self.decom_refAug_ref = self.decom_refAug[0]
        self.decom_refAug_illu = self.decom_refAug[1]

        self.decom_refAlt = self.net_decom(self.ref_alt)
        self.decom_refAlt_ref = self.decom_refAlt[0]
        self.decom_refAlt_illu = self.decom_refAlt[1]

        self.ref_histogram = histcal(self.decom_ref_illu).squeeze(1)
        # self.refRefl_histogram = histcal(self.decom_ref_ref).squeeze(1)
        # self.refAlt_histogram = histcal(self.decom_refAlt_illu).squeeze(1)

        # self.ref_histogram_ = torch.mean(self.ref_histogram, dim=2)
        # self.refAlt_histogram_mean = torch.mean(self.refAlt_histogram, dim=2)

        # self.enhanced_lqRef_illu, self.enhanced_lqRef_refl, self.enhanced_ref_refl = self.net_g(self.decom_lq_illu, self.decom_ref_illu, self.ref_histogram, self.decom_lq_ref, self.decom_ref_ref)

        ##################### HSV histogram #####################
        self.decom_lq_ref_hue = rgb2hsv(self.decom_lq_ref)[:, 0, :, :].unsqueeze(1)
        self.decom_lq_ref_saturation = rgb2hsv(self.decom_lq_ref)[:, 1, :, :].unsqueeze(1)
        self.decom_ref_ref_hue = rgb2hsv(self.decom_ref_ref)[:, 0, :, :].unsqueeze(1)
        self.decom_ref_ref_saturation = rgb2hsv(self.decom_ref_ref)[:, 1, :, :].unsqueeze(1)

        # self.hueVector_lq_ref, _ = self.adaptivePool(torch.mean(self.decom_lq_ref_hue, dim=1)).view(-1).sort(descending=True)
        # self.saturationVector_lq_ref, _ = self.adaptivePool(torch.mean(self.decom_lq_ref_saturation, dim=1)).view(-1).sort(descending=True)
        # self.hueVector_ref_ref, _ = self.adaptivePool(torch.mean(self.decom_ref_ref_hue, dim=1)).view(-1).sort(descending=True)
        # self.saturationVector_ref_ref, _ = self.adaptivePool(torch.mean(self.decom_ref_ref_saturation, dim=1)).view(-1).sort(descending=True)
        # self.cos_similarity_hue = torch.cosine_similarity(self.hueVector_lq_ref, self.hueVector_ref_ref, dim=0)
        # self.cos_similarity_saturation = torch.cosine_similarity(self.saturationVector_lq_ref, self.saturationVector_ref_ref, dim=0)

        self.decom_lq_ref_hueHisto = histcal_tensor(self.decom_lq_ref_hue)
        self.decom_lq_ref_saturationHisto = histcal_tensor(self.decom_lq_ref_saturation)
        self.decom_ref_ref_hueHisto = histcal_tensor(self.decom_ref_ref_hue)
        self.decom_ref_ref_saturationHisto = histcal_tensor(self.decom_ref_ref_saturation)
        self.cos_similarity_hue = torch.cosine_similarity(self.decom_lq_ref_hueHisto, self.decom_ref_ref_hueHisto, dim=1).unsqueeze(1)
        self.cos_similarity_saturation = torch.cosine_similarity(self.decom_lq_ref_saturationHisto, self.decom_ref_ref_saturationHisto, dim=1).unsqueeze(1)

        self.enhanced_lqRef_illu, self.enhanced_lqRef_refl, self.enhanced_ref_refl = self.net_g(self.decom_lq_illu, self.decom_ref_illu, self.ref_histogram, self.decom_lq_ref, self.decom_ref_ref, self.cos_similarity_hue, self.cos_similarity_saturation)
        ####################################################################################

        self.enhanced_lqRef = torch.cat((self.enhanced_lqRef_illu, self.enhanced_lqRef_illu, self.enhanced_lqRef_illu), dim=1) * self.enhanced_lqRef_refl

        # self.enhanced_lqRef_refl_histogram = histcal(self.enhanced_lqRef_refl).squeeze(1)
        self.enhanced_lqRef_illu_histogram = histcal(self.enhanced_lqRef_illu).squeeze(1)

        self.enhanced_lqRef_refl_colorMap = self.enhanced_lqRef_refl / torch.mean(self.enhanced_lqRef_refl, dim=1)
        self.decom_ref_ref_colorMap = self.decom_ref_ref / torch.mean(self.decom_ref_ref, dim=1)

        self.noiseMap_enhanced_lqRef_refl = noiseMap(self.enhanced_lqRef_refl_colorMap)
        self.noiseMap_ref_ref = noiseMap(self.decom_ref_ref_colorMap)

        # view the noiseMap matrix to a vector, sort it from largest to smallest, substract, normalize to [-1, 1], mean
        self.noiseMapVector_lq_ref = self.adaptivePool(torch.mean(self.noiseMap_enhanced_lqRef_refl, dim=1)).view(-1)
        self.noiseMapVector_ref_ref = self.adaptivePool(torch.mean(self.noiseMap_ref_ref, dim=1)).view(-1)

        self.noiseMapVector_lq_ref, self.order_lq_ref = self.noiseMapVector_lq_ref.sort(descending=True)
        self.noiseMapVector_ref_ref, self.order_ref_ref = self.noiseMapVector_ref_ref.sort(descending=True)

        # self.similarity = self.noiseMapVector_ref_ref - self.noiseMapVector_lq_ref
        # self.similarity = (self.similarity - self.similarity.min()) / (self.similarity.max() - self.similarity.min())
        # self.similarity = torch.mean((self.similarity - 0.5) * 2)
        # self.similarity = torch.ones((self.opt['datasets']['train']['batch_size_per_gpu'], 1)).cuda() * self.similarity
        self.cos_similarity = torch.cosine_similarity(self.noiseMapVector_lq_ref, self.noiseMapVector_ref_ref, dim=0)
        self.cos_similarity = (self.cos_similarity - 0.75) * 4
        self.cos_similarity = torch.ones((self.opt['datasets']['train']['batch_size_per_gpu'], 1)).cuda() * self.cos_similarity

        print(self.cos_similarity)

        self.decom_lq_denoisedRef = self.net_denoise(self.enhanced_lqRef_refl, self.cos_similarity)

        # self.enhanced_AugAlt_illu, self.enhanced_AugAlt_refl = self.net_g(self.decom_refAug_illu, self.decom_refAlt_illu, self.refAlt_histogram, self.decom_refAug_ref, self.decom_refAlt_ref)
        # self.enhanced_AugAlt = torch.cat((self.enhanced_AugAlt_illu, self.enhanced_AugAlt_illu, self.enhanced_AugAlt_illu), dim=1) * self.enhanced_AugAlt_refl

        # self.enhanced_AugRef_illu, self.enhanced_AugRef_refl = self.net_g(self.decom_refAug_illu, self.decom_ref_illu, self.ref_histogram, self.decom_refAug_ref, self.decom_ref_ref)
        # self.enhanced_AugRef = torch.cat((self.enhanced_AugRef_illu, self.enhanced_AugRef_illu, self.enhanced_AugRef_illu), dim=1) * self.enhanced_AugRef_refl


    def optimize_parameters(self, current_iter):
        self.log_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.forward(current_iter)
        l_g_total = self.backward_G(current_iter)
        if l_g_total:
            l_g_total.backward()
        # update netG
        self.optimizer_g.step()


        # self.log_dict = self.reduce_loss_dict(self.loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # ref_cri = self.opt['datasets']['val']['ref_cri']
        self.adaptivePool = nn.AdaptiveAvgPool2d((self.opt['noiseMap_block'], self.opt['noiseMap_block']))
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # if ref_cri == 'random':
                index = torch.randint(low=0, high=self.ref.size()[1], size=(1,))[0]
                self.ref = self.ref[:,index,:,:,:]

                # output[0] -> 3 channel reflection; output[1] -> 1 channel illumination
                self.decom_output_low = self.net_decom(self.lq)
                self.decom_output_low_illu = self.decom_output_low[1]
                self.decom_output_low_refl = self.decom_output_low[0]

                # self.decom_output_low_refl += 50 * torch.from_numpy(np.random.normal(loc=0, scale=1, \
                # size=(self.decom_output_low_refl.shape[0], self.decom_output_low_refl.shape[1], self.decom_output_low_refl.shape[2], self.decom_output_low_refl.shape[3]))).cuda() / 255.0

                self.decom_output_ref = self.net_decom(self.ref)
                self.decom_output_ref_illu = self.decom_output_ref[1]
                self.decom_output_ref_refl = self.decom_output_ref[0]

                self.test_ref_histogram = histcal(self.decom_output_ref_illu).squeeze(1)

                # self.enhanced_illu_low, self.enhanced_refl_low, self.enhanced_refl_ref = self.net_g(self.decom_output_low_illu, self.decom_output_ref_illu, self.test_ref_histogram, self.decom_output_low_refl, self.decom_output_ref_refl)

                ##################### HSV histogram #####################
                self.test_decom_lq_ref_hue = rgb2hsv(self.decom_output_low_refl)[:, 0, :, :].unsqueeze(1)
                self.test_decom_lq_ref_saturation = rgb2hsv(self.decom_output_low_refl)[:, 1, :, :].unsqueeze(1)
                self.test_decom_ref_ref_hue = rgb2hsv(self.decom_output_ref_refl)[:, 0, :, :].unsqueeze(1)
                self.test_decom_ref_ref_saturation = rgb2hsv(self.decom_output_ref_refl)[:, 1, :, :].unsqueeze(1)

                # self.test_hueVector_lq_ref, _ = self.adaptivePool(torch.mean(self.test_decom_lq_ref_hue, dim=1)).view(-1).sort(descending=True)
                # self.test_saturationVector_lq_ref, _ = self.adaptivePool(torch.mean(self.test_decom_lq_ref_saturation, dim=1)).view(-1).sort(descending=True)
                # self.test_hueVector_ref_ref, _ = self.adaptivePool(torch.mean(self.test_decom_ref_ref_hue, dim=1)).view(-1).sort(descending=True)
                # self.test_saturationVector_ref_ref, _ = self.adaptivePool(torch.mean(self.test_decom_ref_ref_saturation, dim=1)).view(-1).sort(descending=True)
                # self.test_cos_similarity_hue = torch.cosine_similarity(self.test_hueVector_lq_ref, self.test_hueVector_ref_ref, dim=0)
                # self.test_cos_similarity_saturation = torch.cosine_similarity(self.test_saturationVector_lq_ref, self.test_saturationVector_ref_ref, dim=0)

                self.test_decom_lq_ref_hueHisto = histcal_tensor(self.test_decom_lq_ref_hue)
                self.test_decom_lq_ref_saturationHisto = histcal_tensor(self.test_decom_lq_ref_saturation)
                self.test_decom_ref_ref_hueHisto = histcal_tensor(self.test_decom_ref_ref_hue)
                self.test_decom_ref_ref_saturationHisto = histcal_tensor(self.test_decom_ref_ref_saturation)
                self.test_cos_similarity_hue = torch.cosine_similarity(self.test_decom_lq_ref_hueHisto, self.test_decom_ref_ref_hueHisto, dim=1).unsqueeze(1)
                self.test_cos_similarity_saturation = torch.cosine_similarity(self.test_decom_lq_ref_saturationHisto, self.test_decom_ref_ref_saturationHisto, dim=1).unsqueeze(1)

                self.enhanced_illu_low, self.enhanced_refl_low, self.enhanced_refl_ref = self.net_g(self.decom_output_low_illu, self.decom_output_ref_illu, self.test_ref_histogram, self.decom_output_low_refl, self.decom_output_ref_refl, self.test_cos_similarity_hue, self.test_cos_similarity_saturation)
                ####################################################################################
                self.test_enhancedRefl_colorMap_low = self.enhanced_refl_low / torch.mean(self.enhanced_refl_low, dim=1)
                self.test_reflection_colorMap_ref = self.decom_output_ref_refl / torch.mean(self.decom_output_ref_refl, dim=1)

                self.noiseMap_output_lq = noiseMap(self.test_enhancedRefl_colorMap_low)
                self.noiseMap_output_ref = noiseMap(self.test_reflection_colorMap_ref)

                # view the noiseMap matrix to a vector, sort it from largest to smallest, substract, normalize to [-1, 1], mean
                self.noiseMapVector_lq = self.adaptivePool(torch.mean(self.noiseMap_output_lq, dim=1)).view(-1)
                self.noiseMapVector_ref = self.adaptivePool(torch.mean(self.noiseMap_output_ref, dim=1)).view(-1)

                self.noiseMapVector_lq, self.order_lq_ref = self.noiseMapVector_lq.sort(descending=True)
                self.noiseMapVector_ref, self.order_ref_ref = self.noiseMapVector_ref.sort(descending=True)

                # self.similarity_test = self.noiseMapVector_ref - self.noiseMapVector_lq
                # self.similarity_test = (self.similarity_test - self.similarity_test.min()) / (self.similarity_test.max() - self.similarity_test.min())
                # self.similarity_test = torch.mean((self.similarity_test - 0.5) * 2)
                self.cos_similarity_test = torch.cosine_similarity(self.noiseMapVector_lq, self.noiseMapVector_ref, dim=0)
                self.cos_similarity_test = (self.cos_similarity_test - 0.75) * 4

                # self.cos_similarity_test = 0.0
                self.cos_similarity_test = torch.ones((1, 1)).cuda() * self.cos_similarity_test

                self.denoisedRefl_low = self.net_denoise(self.enhanced_refl_low, self.cos_similarity_test)

        else:
            self.net_g.eval()
            with torch.no_grad():
                # if ref_cri == 'random':
                index = torch.randint(low=0, high=self.ref.size()[1], size=(1,))[0]
                self.ref = self.ref[:,index,:,:,:]

                # output[0] -> 3 channel reflection; output[1] -> 1 channel illumination
                self.decom_output_low = self.net_decom(self.lq)
                self.decom_output_low_illu = self.decom_output_low[1]
                self.decom_output_low_refl = self.decom_output_low[0]

                # self.decom_output_low_refl += 50 * torch.from_numpy(np.random.normal(loc=0, scale=1, \
                # size=(self.decom_output_low_refl.shape[0], self.decom_output_low_refl.shape[1], self.decom_output_low_refl.shape[2], self.decom_output_low_refl.shape[3]))).cuda() / 255.0

                self.decom_output_ref = self.net_decom(self.ref)
                self.decom_output_ref_illu = self.decom_output_ref[1]
                self.decom_output_ref_refl = self.decom_output_ref[0]

                self.test_ref_histogram = histcal(self.decom_output_ref_illu).squeeze(1)

                # self.enhanced_illu_low, self.enhanced_refl_low, self.enhanced_refl_ref = self.net_g(self.decom_output_low_illu, self.decom_output_ref_illu, self.test_ref_histogram, self.decom_output_low_refl, self.decom_output_ref_refl)

                ##################### HSV histogram #####################
                self.test_decom_lq_ref_hue = rgb2hsv(self.decom_output_low_refl)[:, 0, :, :].unsqueeze(1)
                self.test_decom_lq_ref_saturation = rgb2hsv(self.decom_output_low_refl)[:, 1, :, :].unsqueeze(1)
                self.test_decom_ref_ref_hue = rgb2hsv(self.decom_output_ref_refl)[:, 0, :, :].unsqueeze(1)
                self.test_decom_ref_ref_saturation = rgb2hsv(self.decom_output_ref_refl)[:, 1, :, :].unsqueeze(1)

                # self.test_hueVector_lq_ref, _ = self.adaptivePool(torch.mean(self.test_decom_lq_ref_hue, dim=1)).view(-1).sort(descending=True)
                # self.test_saturationVector_lq_ref, _ = self.adaptivePool(torch.mean(self.test_decom_lq_ref_saturation, dim=1)).view(-1).sort(descending=True)
                # self.test_hueVector_ref_ref, _ = self.adaptivePool(torch.mean(self.test_decom_ref_ref_hue, dim=1)).view(-1).sort(descending=True)
                # self.test_saturationVector_ref_ref, _ = self.adaptivePool(torch.mean(self.test_decom_ref_ref_saturation, dim=1)).view(-1).sort(descending=True)
                # self.test_cos_similarity_hue = torch.cosine_similarity(self.test_hueVector_lq_ref, self.test_hueVector_ref_ref, dim=0)
                # self.test_cos_similarity_saturation = torch.cosine_similarity(self.test_saturationVector_lq_ref, self.test_saturationVector_ref_ref, dim=0)

                self.test_decom_lq_ref_hueHisto = histcal_tensor(self.test_decom_lq_ref_hue)
                self.test_decom_lq_ref_saturationHisto = histcal_tensor(self.test_decom_lq_ref_saturation)
                self.test_decom_ref_ref_hueHisto = histcal_tensor(self.test_decom_ref_ref_hue)
                self.test_decom_ref_ref_saturationHisto = histcal_tensor(self.test_decom_ref_ref_saturation)
                self.test_cos_similarity_hue = torch.cosine_similarity(self.test_decom_lq_ref_hueHisto, self.test_decom_ref_ref_hueHisto, dim=1).unsqueeze(1)
                self.test_cos_similarity_saturation = torch.cosine_similarity(self.test_decom_lq_ref_saturationHisto, self.test_decom_ref_ref_saturationHisto, dim=1).unsqueeze(1)

                self.enhanced_illu_low, self.enhanced_refl_low, self.enhanced_refl_ref = self.net_g(self.decom_output_low_illu, self.decom_output_ref_illu, self.test_ref_histogram, self.decom_output_low_refl, self.decom_output_ref_refl, self.test_cos_similarity_hue, self.test_cos_similarity_saturation)
                ####################################################################################
                self.test_enhancedRefl_colorMap_low = self.enhanced_refl_low / torch.mean(self.enhanced_refl_low, dim=1)
                self.test_reflection_colorMap_ref = self.decom_output_ref_refl / torch.mean(self.decom_output_ref_refl, dim=1)

                self.noiseMap_output_lq = noiseMap(self.test_enhancedRefl_colorMap_low)
                self.noiseMap_output_ref = noiseMap(self.test_reflection_colorMap_ref)

                # view the noiseMap matrix to a vector, sort it from largest to smallest, substract, normalize to [-1, 1], mean
                self.noiseMapVector_lq = self.adaptivePool(torch.mean(self.noiseMap_output_lq, dim=1)).view(-1)
                self.noiseMapVector_ref = self.adaptivePool(torch.mean(self.noiseMap_output_ref, dim=1)).view(-1)

                self.noiseMapVector_lq, self.order_lq_ref = self.noiseMapVector_lq.sort(descending=True)
                self.noiseMapVector_ref, self.order_ref_ref = self.noiseMapVector_ref.sort(descending=True)

                # self.similarity_test = self.noiseMapVector_ref - self.noiseMapVector_lq
                # self.similarity_test = (self.similarity_test - self.similarity_test.min()) / (self.similarity_test.max() - self.similarity_test.min())
                # self.similarity_test = torch.mean((self.similarity_test - 0.5) * 2)
                self.cos_similarity_test = torch.cosine_similarity(self.noiseMapVector_lq, self.noiseMapVector_ref, dim=0)
                self.cos_similarity_test = (self.cos_similarity_test - 0.75) * 4
                self.cos_similarity_test = torch.ones((1, 1)).cuda() * self.cos_similarity_test

                self.denoisedRefl_low = self.net_denoise(self.enhanced_refl_low, self.cos_similarity_test)

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            metric_data = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # print(val_data)
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            illumination_low_img = tensor2img(visuals['illumination_low'])
            reflection_low_img = tensor2img(visuals['reflection_low'])

            illumination_ref_img = tensor2img(visuals['illumination_ref'])
            reflection_ref_img = tensor2img(visuals['reflection_ref'])

            enhancedIllu_low_img = tensor2img(visuals['enhancedIllu_low'])
            enhancedRefl_low_img = tensor2img(visuals['enhancedRefl_low'])

            enhancedReflColorMap_low_img = tensor2img(visuals['colorMap_enhanced_lqRef_refl'])
            reflectionColorMap_ref_img = tensor2img(visuals['colorMap_decom_ref_ref'])

            noiseMap_lq_ref_img = tensor2img(visuals['noiseMap_lq_ref'])
            noiseMap_ref_ref_img = tensor2img(visuals['noiseMap_ref_ref'])

            denoisedRefl_low_img = tensor2img(visuals['denoisedRefl_low'])

            enhanced_low_img = tensor2img(visuals['enhanced_low'])
            denoise_low_img = tensor2img(visuals['denoise_low'])

            # ref_img = tensor2img(visuals['ref'])
            # ref_alt_img = tensor2img(visuals['ref_alt'])
            # enhanced_AugRef_img = tensor2img(visuals['enhanced_AugRef'])
            # ref_aug_img = tensor2img(visuals['ref_aug'])

            gt_img = tensor2img(visuals['gt'])
            ref_img = tensor2img(visuals['ref'])

            metric_data['img'] = denoise_low_img
            metric_data['img2'] = gt_img

            # if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                # gt_img = tensor2img([visuals['lq']])
                # metric_data['img2'] = gt_img
                # del self.gt

            # tentative for out of GPU memory
            # del self.lq
            # del self.decom_output_low
            # del self.decom_output_ref
            # torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_illu_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_illu_low.png')
                    save_img_path_refl_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_refl_low.png')

                    save_img_path_ref = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_ref.png')
                    save_img_path_refl_ref = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_refl_ref.png')
                    save_img_path_illu_ref = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_illu_ref.png')

                    save_img_path_enhancedIllu_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhancedIllu_low.png')
                    save_img_path_enhancedRefl_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhancedRefl_low.png')

                    save_img_path_enhancedReflColorMap_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhancedRefl_colorMap_low.png')
                    save_img_path_reflectionColorMap_ref = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_reflectionColorMap_ref.png')

                    save_img_path_noiseMap_lq_ref = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_noiseMap_lq_ref.png')
                    save_img_path_noiseMap_ref_ref = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_noiseMap_ref_ref.png')

                    save_img_path_denoisedRefl_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_denoisedRefl_low.png')

                    save_img_path_enhanced_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_enhanced_low.png')
                    save_img_path_denoised_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_denoised_low.png')

                    save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.png')

                    # save_img_path_ref = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}_ref.png')
                    # save_img_path_ref_alt = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}_ref_alt.png')
                    # save_img_path_enhanced_AugRef = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}_enhanced_AugRef.png')
                    # save_img_path_ref_aug = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}_ref_aug.png')
                    # save_img_path_enhanced_denoisedReflection_low = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}_enhanced_denoisedReflection_low.png')

                else:
                    if self.opt['val']['suffix']:
                        save_img_path_illu_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_illu_low.png')
                        save_img_path_refl_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_refl_low.png')

                        save_img_path_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_ref.png')
                        save_img_path_refl_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_refl_ref.png')
                        save_img_path_illu_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_illu_ref.png')

                        save_img_path_enhancedIllu_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_enhancedIllu_low.png')
                        save_img_path_enhancedRefl_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_enhancedRefl_low.png')

                        save_img_path_enhancedReflColorMap_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_enhancedRefl_colorMap_low.png')
                        save_img_path_reflectionColorMap_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_reflectionColorMap_ref.png')

                        save_img_path_noiseMap_lq_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_noiseMap_lq_ref.png')
                        save_img_path_noiseMap_ref_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_noiseMap_ref_ref.png')

                        save_img_path_denoisedRefl_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_denoisedRefl_low.png')

                        save_img_path_enhanced_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_enhanced_low.png')
                        save_img_path_denoised_low = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_denoised_low.png')

                        save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}_gt.png')

                        # save_img_path_ref = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_ref.png')
                        # save_img_path_ref_alt = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_ref_alt.png')
                        # save_img_path_enhanced_AugRef = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_enhanced_AugRef.png')
                        # save_img_path_ref_aug = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_ref_aug.png')
                        # save_img_path_enhanced_denoisedReflection_low = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_enhanced_denoisedReflection_low.png')
                    else:
                        save_img_path_illu_low = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_illu_low.png')
                        save_img_path_refl_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_refl_low.png')

                        save_img_path_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_ref.png')
                        save_img_path_refl_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_refl_ref.png')
                        save_img_path_illu_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_illu_ref.png')

                        save_img_path_enhancedIllu_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhancedIllu_low.png')
                        save_img_path_enhancedRefl_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhancedRefl_low.png')

                        save_img_path_enhancedReflColorMap_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhancedRefl_colorMap_low.png')
                        save_img_path_reflectionColorMap_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_reflectionColorMap_ref.png')

                        save_img_path_noiseMap_lq_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_noiseMap_lq_ref.png')
                        save_img_path_noiseMap_ref_ref = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_noiseMap_ref_ref.png')

                        save_img_path_denoisedRefl_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_denoisedRefl_low.png')

                        save_img_path_enhanced_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_enhanced_low.png')
                        save_img_path_denoised_low = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_denoised_low.png')

                        save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_gt.png')

                        # save_img_path_ref = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_ref.png')
                        # save_img_path_ref_alt = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_ref_alt.png')
                        # save_img_path_enhanced_AugRef = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_enhanced_AugRef.png')
                        # save_img_path_ref_aug = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_ref_aug.png')
                        # save_img_path_enhanced_denoisedReflection_low = osp.join(self.opt['path']['visualization'], img_name,
                        #                          f'{img_name}_{current_iter}_enhanced_denoisedReflection_low.png')


                # imwrite(reflection_low_img, save_img_path_refl_low)
                # imwrite(illumination_low_img, save_img_path_illu_low)

                # imwrite(ref_img, save_img_path_ref)
                # imwrite(reflection_ref_img, save_img_path_refl_ref)
                # imwrite(illumination_ref_img, save_img_path_illu_ref)

                # imwrite(enhancedIllu_low_img, save_img_path_enhancedIllu_low)
                # imwrite(enhancedRefl_low_img, save_img_path_enhancedRefl_low)

                # imwrite(enhancedReflColorMap_low_img, save_img_path_enhancedReflColorMap_low)
                # imwrite(reflectionColorMap_ref_img, save_img_path_reflectionColorMap_ref)

                # imwrite(noiseMap_lq_ref_img, save_img_path_noiseMap_lq_ref)
                # imwrite(noiseMap_ref_ref_img, save_img_path_noiseMap_ref_ref)

                # imwrite(denoisedRefl_low_img, save_img_path_denoisedRefl_low)

                # imwrite(enhanced_low_img, save_img_path_enhanced_low)
                imwrite(denoise_low_img, save_img_path_denoised_low)

                # imwrite(gt_img, save_img_path_gt)

                # imwrite(ref_img, save_img_path_ref)
                # imwrite(ref_alt_img, save_img_path_ref_alt)
                # imwrite(enhanced_AugRef_img, save_img_path_enhanced_AugRef)
                # imwrite(ref_aug_img, save_img_path_ref_aug)

                # imwrite(enhanced_denoised_img, save_img_path_enhanced_denoised_low)
                # imwrite(enhanced_denoised_reflection_img, save_img_path_enhanced_denoisedReflection_low)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict['lq'] = self.lq.detach().cpu()
        self.decom_low_visual_ref = self.decom_output_low_refl.detach().cpu()
        self.decom_low_visual_ill = torch.cat((self.decom_output_low_illu, self.decom_output_low_illu, self.decom_output_low_illu), dim=1).detach().cpu()
        out_dict['reflection_low'] = self.decom_low_visual_ref
        out_dict['illumination_low'] = self.decom_low_visual_ill

        out_dict['ref'] = self.ref.detach().cpu()
        self.decom_ref_visual_ref = self.decom_output_ref_refl.detach().cpu()
        self.decom_ref_visual_ill = torch.cat((self.decom_output_ref_illu, self.decom_output_ref_illu, self.decom_output_ref_illu), dim=1).detach().cpu()
        out_dict['reflection_ref'] = self.decom_ref_visual_ref
        out_dict['illumination_ref'] = self.decom_ref_visual_ill

        self.enhancedIllu_low = torch.cat((self.enhanced_illu_low, self.enhanced_illu_low, self.enhanced_illu_low), dim=1).detach().cpu()
        self.enhancedRefl_low = self.enhanced_refl_low.detach().cpu()
        out_dict['enhancedIllu_low'] = self.enhancedIllu_low
        out_dict['enhancedRefl_low'] = self.enhancedRefl_low

        out_dict['colorMap_enhanced_lqRef_refl'] = self.test_enhancedRefl_colorMap_low.detach().cpu()
        out_dict['colorMap_decom_ref_ref'] = self.test_reflection_colorMap_ref.detach().cpu()

        out_dict['noiseMap_lq_ref'] = self.noiseMap_output_lq
        out_dict['noiseMap_ref_ref'] = self.noiseMap_output_ref

        out_dict['denoisedRefl_low'] = self.denoisedRefl_low.detach().cpu()

        out_dict['enhanced_low'] = self.enhancedIllu_low * self.enhancedRefl_low
        out_dict['denoise_low'] = self.enhancedIllu_low * self.denoisedRefl_low.detach().cpu()

        # out_dict['ref'] = self.ref
        # out_dict['ref_alt'] = self.ref_alt
        # out_dict['enhanced_AugRef'] = self.enhanced_AugRef
        # out_dict['ref_aug'] = self.ref_aug

        out_dict['gt'] = self.gt

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

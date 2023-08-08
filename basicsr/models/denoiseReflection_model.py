import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class DenoiseReflectionModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DenoiseReflectionModel, self).__init__(opt)

        # define network
        self.net_decom = build_network(opt['network_decom'])
        self.net_decom = self.model_to_device(self.net_decom)
        self.print_network(self.net_decom)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path_decom = self.opt['path'].get('pretrain_network_decom', None)
        if load_path_decom is not None:
            param_key = self.opt['path'].get('param_key_decom', 'params')
            self.load_network(self.net_decom, load_path_decom, self.opt['path'].get('strict_load_decom', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

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
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.reflection = self.net_decom(self.lq)[0]
        self.img_brightness = torch.max(self.lq, axis=1)[0].unsqueeze(dim=1)
        self.noise1 = (torch.normal(0, 1, size=self.reflection.shape).cuda() * self.opt['sigma'] * (1 - self.img_brightness) / 255.0) + self.reflection
        self.noise2 = (torch.normal(0, 1, size=self.reflection.shape).cuda() * self.opt['sigma'] * (1 - self.img_brightness) / 255.0) + self.reflection

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.control = torch.ones((self.opt['datasets']['train']['batch_size_per_gpu'] * self.opt['num_gpu'], 1)).cuda() * self.opt['control']
        self.output_lq = self.net_g(self.noise1, self.control)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_lq, self.noise2)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            self.control_test = torch.ones((1, 1)).cuda() * self.opt['control']
            print(self.control_test)
            with torch.no_grad():
                self.output = self.net_g_ema(self.noise1, self.control_test)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.control_test = torch.ones((1, 1)).cuda() * self.opt['control']
                print(self.control_test)
                self.output = self.net_g(self.noise1, self.control_test)
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
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # [0-255]
            denoised_img = tensor2img([visuals['result']])
            reflection_img = tensor2img(visuals['reflection'])
            noise1_img = tensor2img(visuals['noise1'])
            noise2_img = tensor2img(visuals['noise2'])
            metric_data['img'] = denoised_img
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path_denoised = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_denoised.png')
                    save_img_path_reflection = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_reflection.png')
                    save_img_path_lq1 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_lq1.png')
                    save_img_path_lq2 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_lq2.png')
                    save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path_denoised = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_denoised.png')
                        save_img_path_reflection = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_reflection.png')
                        save_img_path_lq1 = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq1.png')
                        save_img_path_lq2 = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq2.png')
                        save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path_denoised = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_denoised.png')
                        save_img_path_reflection = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_reflection.png')
                        save_img_path_lq1 = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq1.png')
                        save_img_path_lq2 = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_lq2.png')
                        save_img_path_gt = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                imwrite(denoised_img, save_img_path_denoised)
                imwrite(reflection_img, save_img_path_reflection)
                imwrite(noise1_img, save_img_path_lq1)
                imwrite(noise2_img, save_img_path_lq2)
                imwrite(gt_img, save_img_path_gt)

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
        out_dict['reflection'] = self.reflection.detach().cpu()
        out_dict['noise1'] = self.noise1.detach().cpu()
        out_dict['noise2'] = self.noise2.detach().cpu()
        out_dict['result'] = self.output[0].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

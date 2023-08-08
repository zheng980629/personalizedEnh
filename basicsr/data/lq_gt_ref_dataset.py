import numpy as np
import torch
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, \
paired_paths_from_meta_info_file, get_image_paths, read_img, channel_convert, augment
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LQGT_ref_dataset(data.Dataset):
    def __init__(self, opt):
        super(LQGT_ref_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_ref, self.paths_GT = None, None, None
        self.sizes_LQ, self.paths_ref, self.sizes_GT = None, None, None
        self.LQ_env, self.ref_env, self.GT_env = None, None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_ref, self.sizes_ref = get_image_paths(self.data_type, opt['dataroot_ref'])
        self.paths_LQ, self.sizes_LQ = get_image_paths(self.data_type, opt['dataroot_LQ'])
        if self.paths_LQ and self.paths_GT and self.paths_ref:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'LQ and GT datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):
        GT_path, ref_path, LQ_path = None, None, None
        # reference 1
        seed = np.random.randint(0, len(self.paths_ref))
        # reference 2
        seed_alt = np.random.randint(0, len(self.paths_ref))
        while seed == index:
            seed = np.random.randint(0, len(self.paths_ref))
        while seed_alt==index or seed_alt==seed:
            seed_alt = np.random.randint(0, len(self.paths_ref))
            
        if self.paths_GT:
            GT_path = self.paths_GT[index]
        ref_path = self.paths_ref[seed]
        ref_path_alt = self.paths_ref[seed_alt]
        LQ_path = self.paths_LQ[index]
        if self.paths_GT:
            img_GT = read_img(self.GT_env, GT_path)
        img_ref = read_img(self.ref_env, ref_path)
        img_ref_alt = read_img(self.ref_env, ref_path_alt)
        img_LQ = read_img(self.LQ_env, LQ_path)
        if self.opt['color']:
            if self.paths_GT:
                img_GT = channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_ref = channel_convert(img_ref.shape[2], self.opt['color'], [img_ref])[0]
            img_ref_alt = channel_convert(img_ref_alt.shape[2], self.opt['color'], [img_ref_alt])[0]
            img_LQ = channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
        if img_LQ.shape[0]>600 or img_LQ.shape[1] > 600:
            aspect_ratio = img_LQ.shape[0]/img_LQ.shape[1]
            if aspect_ratio>1:
                h = 600
                w = int(h/aspect_ratio)
            else:
                w = 600
                h = int(w*aspect_ratio)
            img_LQ = cv2.resize(img_LQ, (h, w), interpolation=cv2.INTER_CUBIC)
        if img_ref.shape[0]>600 or img_ref.shape[1] > 600:
            aspect_ratio = img_ref.shape[0]/img_ref.shape[1]
            if aspect_ratio>1:
                h = 600
                w = int(h/aspect_ratio)
            else:
                w = 600
                h = int(w*aspect_ratio)
            img_ref = cv2.resize(img_ref, (h, w), interpolation=cv2.INTER_CUBIC)
        if img_ref_alt.shape[0]>600 or img_ref_alt.shape[1] > 600:
            aspect_ratio = img_ref_alt.shape[0]/img_ref_alt.shape[1]
            if aspect_ratio>1:
                h = 600
                w = int(h/aspect_ratio)
            else:
                w = 600
                h = int(w*aspect_ratio)
            img_ref_alt = cv2.resize(img_ref_alt, (h, w), interpolation=cv2.INTER_CUBIC)
            
        # reshape
        if self.opt['resize_to_500']:
            if self.paths_GT:
                img_GT = cv2.resize(img_GT, (500, 500), interpolation=cv2.INTER_CUBIC)
            img_LQ = cv2.resize(img_LQ, (500, 500), interpolation=cv2.INTER_CUBIC)
            img_ref = cv2.resize(img_ref, (500, 500), interpolation=cv2.INTER_CUBIC)
            img_ref_alt = cv2.resize(img_ref_alt, (500, 500), interpolation=cv2.INTER_CUBIC)

        # random flipping
        # img_ref, img_ref_alt = augment([img_ref, img_ref_alt], hflip=True, rot=True)
        if self.paths_GT:
            img_LQ, img_GT, img_ref, img_ref_alt = augment([img_LQ, img_GT, img_ref, img_ref_alt], hflip=True, rot=True)
        else:
            img_LQ, img_ref, img_ref_alt = augment([img_LQ, img_ref, img_ref_alt], hflip=True, rot=True)[0]
            
        # BGR to RGB
        if img_LQ.shape[2] == 3:
            if self.paths_GT:
                img_GT = img_GT[:, :, [2, 1, 0]]
            img_ref = img_ref[:, :, [2, 1, 0]]
            img_ref_alt = img_ref_alt[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        # HWC to CHW, numpy to tensor
        H, W, _ = img_LQ.shape
        if self.paths_GT:
            img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref, (2, 0, 1)))).float()
        img_ref_alt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_ref_alt, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
            
        if self.paths_GT:
            return {'lq': img_LQ, 'ref': img_ref, 'ref_alt': img_ref_alt, 'gt': img_GT, 'lq_path': LQ_path, 'ref_path': ref_path, 'ref_path_alt': ref_path_alt, 'GT_path': GT_path}
        else:
            return {'lq': img_LQ, 'ref': img_ref, 'ref_alt': img_ref_alt, 'lq_path': LQ_path, 'ref_path': ref_path, 'ref_path_alt': ref_path_alt}

    def __len__(self):
        return len(self.paths_LQ)
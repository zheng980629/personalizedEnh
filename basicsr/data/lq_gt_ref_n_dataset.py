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
class LQGT_ref_n_dataset(data.Dataset):
    def __init__(self, opt):
        super(LQGT_ref_n_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_ref, self.paths_GT = None, None, None
        self.sizes_LQ, self.paths_ref, self.sizes_GT = None, None, None
        self.LQ_env, self.ref_env, self.GT_env = None, None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_ref, self.sizes_ref = get_image_paths(self.data_type, opt['dataroot_ref'])
        self.paths_LQ, self.sizes_LQ = get_image_paths(self.data_type, opt['dataroot_LQ'])

        self.n_refs = opt['n_refs'] if opt['n_refs'] else 20
        if self.n_refs > len(self.paths_ref):
            self.n_refs = len(self.paths_ref)

        if self.paths_LQ and self.paths_GT and self.paths_ref:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

        # refs_path = map(lambda x: self.paths_ref[x], [0,1,2,3])
        # print(refs_path)

    def __getitem__(self, index):
        GT_path, ref_path, LQ_path = None, None, None

        seed_list = []
        if self.n_refs < len(self.paths_ref):
            for i in range(self.n_refs):
                ##################### random N reference images #####################
                seed = np.random.randint(0, len(self.paths_ref))
                while (seed == index) or (seed in seed_list):
                    seed = np.random.randint(0, len(self.paths_ref))
                seed_list.append(seed)
        else:
            seed_list = np.random.permutation(self.n_refs)

        # get LQ and GT image
        if self.paths_GT:
            GT_path = self.paths_GT[index]

        LQ_path = self.paths_LQ[index]
        img_GT = read_img(self.GT_env, GT_path) if self.paths_GT else None
        img_LQ = read_img(self.LQ_env, LQ_path)
        # get a list of ref images
        refs_path = list(map(lambda x: self.paths_ref[x], seed_list))

        print(refs_path)

        img_refs = list(map(lambda x: read_img(self.ref_env, x), refs_path))
        if self.opt['resize_ref_to_500']:
            img_refs = list(map(lambda x: cv2.resize(x, (500, 500), interpolation=cv2.INTER_CUBIC), img_refs))
        if self.opt['color']:
            img_GT = channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]  if self.paths_GT else None
            img_refs = list(map(lambda x: channel_convert(x.shape[2], self.opt['color'], [x])[0], img_refs))
            img_LQ = channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]  if self.paths_GT else None
            img_refs = list(map(lambda x: x[:, :, [2, 1, 0]], img_refs))
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        H, W, _ = img_LQ.shape
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()  if self.paths_GT else None
        img_refs = list(map(lambda x: torch.from_numpy(np.ascontiguousarray(np.transpose(x, (2, 0, 1)))).float(), img_refs))
        img_refs = torch.stack(img_refs, dim=0)
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if refs_path is None:
            refs_path = GT_path
        if self.paths_GT:
            return {'lq': img_LQ, 'ref': img_refs, 'gt': img_GT, 'lq_path': LQ_path, 'ref_path': refs_path, 'GT_path': GT_path}
        else:
            return {'lq': img_LQ, 'ref': img_refs, 'lq_path': LQ_path, 'ref_path': refs_path}

    def __len__(self):
        return len(self.paths_LQ)
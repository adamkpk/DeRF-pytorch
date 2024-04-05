import torch
from tqdm import tqdm

from datasets import dataset_dict
from config import DATASET_NAME, DATASET_SIZE_DICT

import pickle
import numpy as np


def load_data(split='train'):
    if DATASET_NAME not in dataset_dict.keys():
        raise Exception('Invalid dataset name')

    dir_dict = {'blender': './../data/nerf_synthetic/lego',
                'llff': './../data/nerf_llff_data/fern'}

    kwargs = {'root_dir': dir_dict[DATASET_NAME],
              'split': split,
              'img_wh': DATASET_SIZE_DICT[DATASET_NAME]}

    if DATASET_NAME == 'llff':
        kwargs['spheric_poses'] = True

    dataset = dataset_dict[DATASET_NAME](**kwargs)

    out_set = torch.zeros((1, 1))

    if split == 'train':
        print('Generating training dataset...')

        out_set = torch.concatenate((dataset[:]['rays'][:, :6], dataset[:]['rgbs']), dim=-1)

        # Memorial: terrible working version of above vectorized image flattener
        # out_set = torch.zeros((len(dataset), 9))
        #
        # for i in tqdm(range(len(dataset))):
        #     sample = dataset[i]
        #
        #     ray_org_dir = torch.Tensor(sample['rays'][:6])
        #     gt_pixel_color = torch.Tensor(sample['rgbs'])
        #
        #     out_set[i] = torch.cat((ray_org_dir, gt_pixel_color))

    elif split == 'test':
        print('Generating testing dataset...')

        out_set = torch.zeros((len(dataset) * len(dataset[0]['rays']), 6))

        for i in tqdm(range(len(dataset))):
            test_image = dataset[i]

            flatten_start = i * len(dataset[0]['rays'])
            flatten_end = (i + 1) * len(dataset[0]['rays'])

            out_set[flatten_start:flatten_end] = test_image['rays'][:, :6]

            # Memorial: terrible not-working version of above vectorized image flattener
            # for j in range(len(test_image['rays'])):
            #     ray_org_dir = torch.Tensor(test_image['rays'][j][0:6])
            #     # gt_pixel_color = torch.Tensor(group['rgbs'][j])
            #
            #     # out_set[len(dataset) * i + j] = torch.cat((ray_org_dir, gt_pixel_color))
            #     out_set[len(dataset) * i + j] = ray_org_dir

    return out_set, dataset[0]['near'], dataset[0]['far']

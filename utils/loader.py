import torch
import numpy as np
from tqdm import tqdm

from datasets import dataset_dict
from config import DATASET_NAME, DATASET_TYPE, DATASET_SIZE_DICT


def load_data(split='train'):
    if DATASET_NAME not in dataset_dict.keys():
        raise Exception('Invalid dataset name')

    dir_dict = {'blender': f'./../data/nerf_synthetic/{DATASET_TYPE}',
                'llff': f'./../data/nerf_llff_data/{DATASET_TYPE}'}

    kwargs = {'root_dir': dir_dict[DATASET_NAME],
              'split': split,
              'img_wh': DATASET_SIZE_DICT[DATASET_NAME]}

    if DATASET_NAME == 'llff':
        kwargs['spheric_poses'] = False  # was true

    dataset = dataset_dict[DATASET_NAME](**kwargs)

    out_set = torch.zeros((1, 1))

    if split == 'train':
        print('Generating training dataset...')

        out_set = torch.concatenate((dataset[:]['rays'][:, :6], dataset[:]['rgbs']), dim=-1)

    elif split == 'test':
        print('Generating testing dataset...')

        out_set = torch.zeros((len(dataset) * len(dataset[0]['rays']), 6))

        for i in tqdm(range(len(dataset))):
            test_image = dataset[i]

            flatten_start = i * len(dataset[0]['rays'])
            flatten_end = (i + 1) * len(dataset[0]['rays'])

            out_set[flatten_start:flatten_end] = test_image['rays'][:, :6]

    return out_set, dataset[0]['near'], dataset[0]['far']


def compute_bounding_box(rays, near, far):
    origins = np.array(rays[:, :3])
    directions = np.array(rays[:, 3:])

    near_intersections = origins + near * directions
    far_intersections = origins + far * directions

    all_intersection_points = np.concatenate([near_intersections, far_intersections], axis=0)

    min_corner = np.min(all_intersection_points, axis=0)
    max_corner = np.max(all_intersection_points, axis=0)

    return min_corner, max_corner

import os
import re
import pickle
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from scipy import ndimage

from config import (DEVICE,
                    TEST_ALL_EPOCHS,
                    NUM_BINS,
                    DATASET_NAME,
                    DATASET_TYPE,
                    DATASET_SIZE_DICT,
                    DATASET_TEST_SIZE)

from utils.metrics import (compute_rmse,
                           compute_psnr,
                           compute_ssim,
                           compute_lpips,
                           aggregate_metrics,
                           aggregate_images)

from model.nerf import (sample_ray_positions,
                        evaluate_rays,
                        integrate_ray_color)

from model.derf import (Voronoi,
                        DeRF,
                        partition_samples)


@torch.no_grad()
def test(model_derf, model_voronoi, dataset, near, far, epoch, img_index, bins, height, width, chunk_size=10):
    ray_origins = dataset[img_index * height * width: (img_index + 1) * height * width, :3]
    ray_directions = dataset[img_index * height * width: (img_index + 1) * height * width, 3:6]

    head_positions = model_voronoi.head_positions.detach().cpu().numpy()
    n_heads = head_positions.shape[0]

    data = []   # list of lists of regenerated pixel values for each isolated head
    for _ in range(n_heads):
        data.append([])

    for i in tqdm(range(int(np.ceil(height / chunk_size)))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * width * chunk_size: (i + 1) * width * chunk_size].to(DEVICE)
        ray_directions_ = ray_directions[i * width * chunk_size: (i + 1) * width * chunk_size].to(DEVICE)

        x, delta = sample_ray_positions(ray_origins_, ray_directions_, near, far, bins)

        region_indices = torch.argmax(model_voronoi(x), dim=-1)

        for head_index, head in enumerate(model_derf.heads):
            region_mask = torch.zeros_like(region_indices).to(DEVICE)
            region_mask[region_indices == head_index] = 1

            if torch.any(region_mask):
                head_sigma, head_colors = evaluate_rays(head, ray_directions_, bins, x, region_mask)
                regenerated_px_values = integrate_ray_color(head_sigma, delta, head_colors)
            else:  # if no samples in the batch are evaluated by head {head_index}
                # ones for white background
                regenerated_px_values = torch.ones(ray_origins_.shape[0], 3).to(DEVICE)

            data[head_index].append(regenerated_px_values)

    # Save the images for each head
    for i in range(n_heads):
        img = torch.cat(data[i]).data.cpu().numpy().reshape(height, width, 3)

        img_path = os.path.join(results_dir, f'e{epoch}_img{img_index}_head{i}.png')
        os.makedirs(results_dir, exist_ok=True)
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)


def testing_loop():
    with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    testing_dataset = full_dataset[1]

    near = full_dataset[2]
    far = full_dataset[3]

    checkpoint_root = './../checkpoints/derf'
    checkpoint_dir_heads = f'{checkpoint_root}/heads/{DATASET_NAME}/{DATASET_TYPE}'
    checkpoint_dir_voronoi = f'{checkpoint_root}/voronoi/{DATASET_NAME}/{DATASET_TYPE}'

    checkpoint_contents = os.listdir(f'{checkpoint_dir_heads}')

    if not TEST_ALL_EPOCHS:
        # Find epoch of highest number

        highest_epoch = -1
        highest_checkpoint = checkpoint_contents[0]

        for checkpoint in checkpoint_contents:
            match = re.match(r'e(\d+)\.pt', checkpoint)
            epoch = int(match.group(1)) if match else -1

            if epoch > highest_epoch:
                highest_epoch = epoch
                highest_checkpoint = checkpoint

        checkpoint_contents = [highest_checkpoint]

    print('Rendering test views for each DeRF head for the following epochs:\n', checkpoint_contents)

    state_voronoi = torch.load(f'{checkpoint_dir_voronoi}/e0.pt')

    model_voronoi = Voronoi(state_voronoi['head_count'], state_voronoi['bounding_box'])

    model_voronoi.load_state_dict(state_voronoi['state_dict'])

    for param in model_voronoi.parameters():
        param.requires_grad = False

    for checkpoint in checkpoint_contents:
        model_derf = DeRF(model_voronoi.head_positions)

        head_state_dicts = torch.load(f'{checkpoint_dir_heads}/{checkpoint}')

        for head, state_dict in zip(model_derf.heads, head_state_dicts):
            head.load_state_dict(state_dict)

        match = re.match(r'e(\d+)\.pt', checkpoint)
        epoch = int(match.group(1)) if match else -1

        for img_index in range(DATASET_TEST_SIZE[DATASET_NAME][DATASET_TYPE]):
            test(model_derf, model_voronoi, testing_dataset, near, far, epoch, img_index, NUM_BINS['fine'],
                 DATASET_SIZE_DICT[DATASET_NAME][1], DATASET_SIZE_DICT[DATASET_NAME][0])


if __name__ == '__main__':
    run_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(f'./../results/derf_heads/{DATASET_NAME}/{DATASET_TYPE}', run_date)
    testing_loop()
    print('Testing complete, all images and metrics saved.')

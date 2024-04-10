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
                    BINS_FINE,
                    DATASET_NAME,
                    DATASET_TYPE,
                    DATASET_SIZE_DICT,
                    DATASET_TEST_SIZE)

from utils.metrics import (compute_rmse,
                           compute_psnr,
                           compute_ssim,
                           compute_lpips)

from model.nerf import NeRF, render_rays


@torch.no_grad()
def test(model, dataset, near, far, epoch, img_index, bins, height, width, chunk_size=10):
    ray_origins = dataset[img_index * height * width: (img_index + 1) * height * width, :3]
    ray_directions = dataset[img_index * height * width: (img_index + 1) * height * width, 3:6]

    data = []   # list of regenerated pixel values

    for i in tqdm(range(int(np.ceil(height / chunk_size)))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * width * chunk_size: (i + 1) * width * chunk_size].to(DEVICE)
        ray_directions_ = ray_directions[i * width * chunk_size: (i + 1) * width * chunk_size].to(DEVICE)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, near, far, bins)
        data.append(regenerated_px_values)

    # Save the image

    img = torch.cat(data).data.cpu().numpy().reshape(height, width, 3)

    img_dir = os.path.join(f'./../results/{DATASET_NAME}/{DATASET_TYPE}', run_date)
    img_path = os.path.join(img_dir, f'e{epoch}_img{img_index}.png')
    os.makedirs(img_dir, exist_ok=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(img_path)

    # Save the metrics

    target_paths = {
        'blender': f'./../data/nerf_synthetic/lego/test/r_{img_index}.png',
        'llff': f'./../data/nerf_llff_data/{DATASET_TYPE}/images_8/image{img_index:03d}.png'
    }

    prediction = np.array(Image.open(img_path))
    target = np.array(Image.open(target_paths[DATASET_NAME]))

    # Blender target images are 800x800x4, need to convert to 400x400x3
    if DATASET_NAME == 'blender':
        target = ndimage.zoom(target[..., :3], (0.5, 0.5, 1))

    metrics = {
        'rmse': compute_rmse(prediction, target),
        'psnr': compute_psnr(prediction, target),
        'ssim': compute_ssim(prediction, target),
        'lpips': float(compute_lpips(prediction, target)[0][0][0][0])
    }

    metrics_path = os.path.join(img_dir, f'e{epoch}_metrics{img_index}.json')

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


def testing_loop():
    with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    testing_dataset = full_dataset[1]

    near = full_dataset[2]
    far = full_dataset[3]

    checkpoint_dir = f'./../checkpoints/{DATASET_NAME}/{DATASET_TYPE}'
    checkpoint_contents = os.listdir(checkpoint_dir)

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

    print('Testing the following epochs:\n', checkpoint_contents)

    for checkpoint in checkpoint_contents:
        model = NeRF().to(DEVICE)

        model.load_state_dict(torch.load(f'{checkpoint_dir}/{checkpoint}'))
        # model.eval()

        match = re.match(r'e(\d+)\.pt', checkpoint)
        epoch = int(match.group(1)) if match else -1

        for img_index in range(DATASET_TEST_SIZE[DATASET_NAME][DATASET_TYPE]):
            test(model, testing_dataset, near, far, epoch, img_index, BINS_FINE,
                 DATASET_SIZE_DICT[DATASET_NAME][1], DATASET_SIZE_DICT[DATASET_NAME][0])


if __name__ == '__main__':
    run_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    testing_loop()
    print('Testing complete, all images and metrics saved.')

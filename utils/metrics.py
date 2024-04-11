import os
import re
import json
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
import lpips


def compute_rmse(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())


def compute_psnr(prediction, target):
    mse = np.mean((prediction - target) ** 2)

    if mse == 0:
        return float('inf')

    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(prediction, target):
    pred_gray = np.array(Image.fromarray(prediction).convert('L'))
    targ_gray = np.array(Image.fromarray(target).convert('L'))
    return structural_similarity(pred_gray, targ_gray)


def compute_lpips(prediction, target):
    # input is numpy arrays in format (width, height, channel)
    # lpips expects torch tensors in format (batch, channel, width, height)

    prediction = torch.Tensor(prediction).permute(2, 0, 1).unsqueeze(0)
    target = torch.Tensor(target).permute(2, 0, 1).unsqueeze(0)

    print(prediction.shape, target.shape)

    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn.forward(prediction, target)


def aggregate_metrics(results_dir, epoch):
    results_contents = os.listdir(results_dir)

    metrics_mean = {
        'rmse': 0,
        'psnr': 0,
        'ssim': 0,
        'lpips': 0
    }

    metrics_count = 0

    for result in results_contents:
        match = re.match(rf'e{epoch}_metrics\d+\.json', result)

        if match is None:
            continue

        metrics_count += 1

        metrics_path = os.path.join(results_dir, result)

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

            for metric in metrics_mean.keys():
                metrics_mean[metric] += metrics[metric]

    for metric in metrics_mean.keys():
        metrics_mean[metric] /= metrics_count

    metrics_path = os.path.join(results_dir, f'e{epoch}_metrics_mean.json')

    with open(metrics_path, "w") as f:
        json.dump(metrics_mean, f)

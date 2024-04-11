import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt

from config import (DEVICE,
                    BINS_COARSE,
                    BINS_FINE,
                    HIDDEN_UNITS,
                    HEAD_COUNT,
                    DATASET_NAME,
                    DATASET_TYPE,
                    TRAINING_ACCELERATION,
                    DATASET_EPOCHS,
                    DATASET_MILESTONES,
                    DATASET_EPOCHS_COARSE)

import model.nerf as nerf
import model.nerf_train as nerf_train
from model.derf import Voronoi, DeRF, ray_contributions


def training_loop():
    with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    training_dataset = full_dataset[0]

    near = full_dataset[2]
    far = full_dataset[3]

    bounding_box = full_dataset[4]

    print('Training coarse NeRF approximation')

    coarse_nerf = nerf.NeRF().to(DEVICE)
    coarse_nerf_optimizer = torch.optim.Adam(coarse_nerf.parameters(), lr=5e-4)
    nerf_data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    nerf_train.train(coarse_nerf, coarse_nerf_optimizer, None, nerf_data_loader,
                     near, far, int(DATASET_EPOCHS_COARSE[DATASET_NAME]), BINS_COARSE)

    # coarse_nerf.load_state_dict(torch.load('./../checkpoints/coarse/blender/lego/e0.pt'))

    print('Training Voronoi decomposition')

    model_voronoi = Voronoi(HEAD_COUNT, bounding_box).to(DEVICE)
    voronoi_optimizer = torch.optim.Adam(model_voronoi.parameters(), lr=5e-4)
    voronoi_data_loader = DataLoader(training_dataset, batch_size=16384, shuffle=True)

    head_origins = model_voronoi.heads.detach().cpu().numpy().copy()

    voronoi_epochs = 1

    for j in range(voronoi_epochs):
        i = 0

        for batch in tqdm(voronoi_data_loader):
            with torch.no_grad():
                # Coarsely sample density across ALL rays to train voronoi model

                ray_origins = batch[:, :3].to(DEVICE)
                ray_directions = batch[:, 3:6].to(DEVICE)

                x, delta = nerf.sample_ray_positions(ray_origins, ray_directions, near, far, BINS_COARSE)

                sigma, _ = nerf.evaluate_rays(coarse_nerf, ray_directions, BINS_COARSE, x)

            # Softmax Voronoi weights along each sampled ray position (coarse/discrete integration)

            voronoi_weights = model_voronoi(x)

            contributions = ray_contributions(sigma, delta, voronoi_weights)

            with torch.no_grad():
                """
                Minimum possible loss for the batch - as discussed in the paper, the norm of the expected value of the
                ray contrbutions to each head achieves the minimum possible loss when the contributions are uniform
                across all heads. Since the sum of the expected values of the contributions is constant, a vector with
                all elements equal to the mean of the mean of the contributions represents the floor of the loss
                function (i.e. the minimum possible loss per batch). We subtract this from the loss (as subtracting
                by a constant does not affect gradients) for better loss interpretability.
                """
                min_uniform_loss = torch.norm(torch.full_like(contributions[0], float(torch.mean(contributions))))
                min_uniform_loss.requires_grad = False

                if i % 20 == 0:
                    # print('min Loss:', min_uniform_loss.item())
                    # print(torch.full_like(contributions[0], float(torch.mean(contributions))).detach().cpu().numpy())
                    woo = torch.mean(contributions, dim=0).detach().cpu().numpy()
                    print(f'Mean: {np.mean(woo)}, Std: {np.std(woo)}')

            loss = torch.norm(torch.mean(contributions, dim=0)) - min_uniform_loss

            loss.backward()
            voronoi_optimizer.step()

            if i % 20 == 0:
                print('Loss:', loss.detach().cpu())

            i += 1

    head_news = model_voronoi.heads.detach().cpu().numpy()

    print('Plotting Voronoi decomposition')

    mpl.use('TkAgg')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(head_origins[:, 0], head_origins[:, 1], head_origins[:, 2], c='b', marker='x')
    ax.scatter(head_news[:, 0], head_news[:, 1], head_news[:, 2], c='r', marker='o')
    plt.show()

    print('Training DeRF with learned head positions')


if __name__ == '__main__':
    training_loop()
    print('Training complete, all epochs saved.')

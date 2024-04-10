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

    coarse_nerf = nerf.NeRF(hidden_dim=HIDDEN_UNITS['head']).to(DEVICE)
    coarse_nerf_optimizer = torch.optim.Adam(coarse_nerf.parameters(), lr=5e-4)
    nerf_data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    # nerf_train.train(coarse_nerf, coarse_nerf_optimizer, None, nerf_data_loader,
    #                  near, far, int(DATASET_EPOCHS_COARSE[DATASET_NAME]), BINS_COARSE)

    coarse_nerf.load_state_dict(torch.load('./../checkpoints/blender/lego/e0.pt'))

    print('Training Voronoi decomposition')

    model_voronoi = Voronoi(HEAD_COUNT, bounding_box).to(DEVICE)
    voronoi_optimizer = torch.optim.Adam(model_voronoi.parameters(), lr=5e-4)
    voronoi_data_loader = DataLoader(training_dataset, batch_size=16384, shuffle=True)

    hs_org = model_voronoi.heads.detach().cpu().numpy().copy()

    epochs = 1

    for j in range(epochs):
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

            # print(torch.min(voronoi_weights), torch.max(voronoi_weights))

            contributions = ray_contributions(sigma, delta, voronoi_weights)

            # print(torch.min(contributions), torch.max(contributions))

            loss = torch.norm(torch.mean(contributions, dim=0))

            loss.backward()
            voronoi_optimizer.step()

            if i % 20 == 0:
                print('Loss:', loss.detach().cpu())

            i += 1

    hs = model_voronoi.heads.detach().cpu().numpy()

    mpl.use('TkAgg')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hs_org[:, 0], hs_org[:, 1], hs_org[:, 2], c='b', marker='x')
    ax.scatter(hs[:, 0], hs[:, 1], hs[:, 2], c='r', marker='o')
    plt.show()


if __name__ == '__main__':
    training_loop()
    print('Training complete, all epochs saved.')

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
                    HEAD_COUNT,
                    DATASET_NAME,
                    DATASET_TYPE,
                    TRAINING_ACCELERATION,
                    DATASET_EPOCHS)

from model.nerf import (NeRF,
                        sample_ray_positions,
                        evaluate_rays,
                        integrate_ray_color)

from model.derf import (Voronoi,
                        DeRF,
                        ray_contributions,
                        partition_samples)


def train_voronoi(model_voronoi, voronoi_optimizer, voronoi_data_loader, coarse_nerf, near, far):
    head_origins = model_voronoi.head_positions.detach().cpu().numpy().copy()

    voronoi_epochs = 1

    for j in range(voronoi_epochs):
        i = 0

        for batch in tqdm(voronoi_data_loader):
            with torch.no_grad():
                # Coarsely sample density across ALL rays to train voronoi model

                ray_origins = batch[:, :3].to(DEVICE)
                ray_directions = batch[:, 3:6].to(DEVICE)

                x, delta = sample_ray_positions(ray_origins, ray_directions, near, far, BINS_COARSE)

                sigma, _ = evaluate_rays(coarse_nerf, ray_directions, BINS_COARSE, x)

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

        checkpoint_dir = f'./../checkpoints/derf/voronoi/{DATASET_NAME}/{DATASET_TYPE}'
        checkpoint_path = os.path.join(checkpoint_dir, f'e{j}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model_voronoi.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

    head_news = model_voronoi.head_positions.detach().cpu().numpy()

    print('Plotting Voronoi decomposition')

    mpl.use('TkAgg')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(head_origins[:, 0], head_origins[:, 1], head_origins[:, 2], c='b', marker='x')
    ax.scatter(head_news[:, 0], head_news[:, 1], head_news[:, 2], c='r', marker='o')
    plt.show()


def train_derf(model_derf, model_voronoi, data_loader, near, far, epochs, bins):
    head_positions = model_voronoi.head_positions.detach().cpu().numpy()

    for i in range(epochs):
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(DEVICE)
            ray_directions = batch[:, 3:6].to(DEVICE)
            ground_truth_px_values = batch[:, 6:].to(DEVICE)

            x, delta = sample_ray_positions(ray_origins, ray_directions, near, far, bins)

            # [batch_size, bins] array of corresponding Voronoi region indices in which each ray sample resides
            region_indices = torch.from_numpy(partition_samples(x.detach().cpu().numpy(), head_positions)).to(DEVICE)

            head_regenerated_px_values = torch.zeros((len(head_positions), batch.shape[0], 3)).to(DEVICE)

            for head_index, head in enumerate(model_derf.heads):
                region_mask = torch.zeros_like(region_indices).to(DEVICE)
                region_mask[region_indices == head_index] = 1

                if not torch.any(region_mask):
                    continue

                # [batch_size, bins], [batch_size, bins, 3]
                sigma, colors = evaluate_rays(head['model'], ray_directions, bins, x, region_mask)

                # [batch_size, 3]
                head_regenerated_px_values[head_index] = integrate_ray_color(sigma, delta, colors)

            # worry later, should be using painter's algorithm here but doing this for quick test to see if it trains
            regenerated_px_values = torch.sum(head_regenerated_px_values, dim=0)

            for head in model_derf.heads:
                head['optimizer'].zero_grad()

            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            loss.backward()

            for head in model_derf.heads:
                head['optimizer'].step()

        for head in model_derf.heads:
            head['scheduler'].step()

        checkpoint_dir = f'./../checkpoints/derf/heads/{DATASET_NAME}/{DATASET_TYPE}'
        checkpoint_path = os.path.join(checkpoint_dir, f'e{i}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save([head['model'].state_dict() for head in model_derf.heads], checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')


def training_loop():
    with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    training_dataset = full_dataset[0]

    near = full_dataset[2]
    far = full_dataset[3]

    bounding_box = full_dataset[4]

    print('Training coarse NeRF approximation')

    model_nerf_coarse = NeRF().to(DEVICE)
    nerf_optimizer = torch.optim.Adam(model_nerf_coarse.parameters(), lr=5e-4)
    nerf_data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    # nerf_train.train(model_nerf_coarse, nerf_optimizer, None, nerf_data_loader,
    #                  near, far, int(DATASET_EPOCHS_COARSE[DATASET_NAME]), BINS_COARSE, 'derf')

    model_nerf_coarse.load_state_dict(torch.load(f'./../checkpoints/derf/coarse/{DATASET_NAME}/{DATASET_TYPE}/e0.pt'))

    print('Training Voronoi decomposition')

    model_voronoi = Voronoi(HEAD_COUNT, bounding_box).to(DEVICE)
    voronoi_optimizer = torch.optim.Adam(model_voronoi.parameters(), lr=5e-4)
    voronoi_data_loader = DataLoader(training_dataset, batch_size=16384, shuffle=True)

    # train_voronoi(model_voronoi, voronoi_optimizer, voronoi_data_loader, model_nerf_coarse, near, far)

    model_voronoi.load_state_dict(torch.load(f'./../checkpoints/derf/voronoi/{DATASET_NAME}/{DATASET_TYPE}/e0.pt'))

    print('Training DeRF with learned head positions')

    model_derf = DeRF(model_voronoi.head_positions)

    derf_data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    # train_derf(model_derf, model_voronoi, derf_data_loader, near, far,
    #            int(DATASET_EPOCHS[DATASET_NAME] / TRAINING_ACCELERATION), BINS_COARSE)

    train_derf(model_derf, model_voronoi, derf_data_loader, near, far, 1, BINS_COARSE)

    # head_state_dicts = torch.load(f'./../checkpoints/derf/heads/{DATASET_NAME}/{DATASET_TYPE}/e0.pt')
    #
    # for head, state_dict in zip(model_derf.heads, head_state_dicts):
    #     head.load_state_dict(state_dict)


if __name__ == '__main__':
    training_loop()
    print('Training complete, all epochs saved.')

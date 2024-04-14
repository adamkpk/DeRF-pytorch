import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import nerf_train

from config import (DEVICE,
                    NUM_BINS,
                    HEAD_COUNT,
                    DATASET_NAME,
                    DATASET_TYPE,
                    TRAINING_ACCELERATION,
                    DATASET_EPOCHS,
                    DATASET_EPOCHS_COARSE,
                    DATASET_MILESTONES)

from model.nerf import (NeRF,
                        sample_ray_positions,
                        evaluate_rays,
                        integrate_ray_color)

from model.derf import (Voronoi,
                        DeRF,
                        ray_contributions,
                        partition_samples)


def train_voronoi(model_voronoi, voronoi_optimizer, voronoi_data_loader, coarse_nerf, near, far):
    checkpoint_dir = f'./../checkpoints/derf/voronoi/{DATASET_NAME}/{DATASET_TYPE}'

    head_origins = model_voronoi.head_positions.detach().cpu().numpy().copy()

    voronoi_epochs = 1

    for j in range(voronoi_epochs):
        iters = 0
        total_iters = len(voronoi_data_loader)

        epoch_losses = []

        for batch in tqdm(voronoi_data_loader):
            with torch.no_grad():
                # Coarsely sample density across ALL rays to train voronoi model

                ray_origins = batch[:, :3].to(DEVICE)
                ray_directions = batch[:, 3:6].to(DEVICE)

                x, delta = sample_ray_positions(ray_origins, ray_directions, near, far, NUM_BINS['coarse'])

                sigma, _ = evaluate_rays(coarse_nerf, ray_directions, NUM_BINS['coarse'], x)

            # Softmax Voronoi weights along each sampled ray position (coarse/discrete integration)

            voronoi_weights = model_voronoi(x)

            # density-weighted head contributions across each rays in batch (see paper section 3.2 / report)
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

            # mean density-weighted head contributions across all rays in batch
            W_phi = torch.mean(contributions, dim=0)

            loss = torch.norm(W_phi) - min_uniform_loss
            epoch_losses.append(loss.item())

            if iters % 20 == 0:
                print(f'Loss: {loss.item():.6f}    Stdev: {np.std(W_phi.detach().cpu().numpy()):.6f}')

            loss.backward()
            voronoi_optimizer.step()

            iters += 1

        checkpoint_path = os.path.join(checkpoint_dir, f'e{j}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'state_dict': model_voronoi.state_dict(),
            'head_count': model_voronoi.head_count,
            'bounding_box': model_voronoi.bounding_box
        }, checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        plt.figure()
        plt.plot(epoch_losses)
        plt.title(f'Voronoi partition loss - Epoch {j}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(checkpoint_dir, f'loss_e{j}.png'))
        plt.close()
        print(f'Saved summary visualization for epoch {j}.')

    head_news = model_voronoi.head_positions.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(head_origins[:, 0], head_origins[:, 1], head_origins[:, 2], c='b', marker='x', label='Initial')
    ax.scatter(head_news[:, 0], head_news[:, 1], head_news[:, 2], c='r', marker='o', label='Learned')
    plt.title(f'Voronoi partition')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, f'voronoi.png'))
    plt.close()
    print('Saved Voronoi decomposition')


def train_derf(model_derf, model_voronoi, optimizer, scheduler, data_loader, near, far, epochs, bins):
    # Freeze Voronoi model weights for safety / clarity that no further training should take place on it
    for param in model_voronoi.parameters():
        param.requires_grad = False

    iters = 0
    for i in range(epochs):
        print(f'Training DeRF {DATASET_TYPE}. Epoch: {i}')
        epoch_losses = []
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(DEVICE)
            ray_directions = batch[:, 3:6].to(DEVICE)
            ground_truth_px_values = batch[:, 6:].to(DEVICE)

            x, delta = sample_ray_positions(ray_origins, ray_directions, near, far, bins)

            # get hard Voronoi cell regions via argmax soft forward pass
            decomposition_weights = model_voronoi(x)
            region_indices = torch.argmax(decomposition_weights, dim=-1)

            sigma = torch.zeros(batch.shape[0], bins).to(DEVICE)
            colors = torch.zeros(batch.shape[0], bins, 3).to(DEVICE)

            for head_index, head in enumerate(model_derf.heads):
                region_mask = torch.zeros_like(region_indices).to(DEVICE)
                region_mask[region_indices == head_index] = 1

                if torch.any(region_mask):
                    # [batch_size, bins], [batch_size, bins, 3]
                    head_sigma, head_colors = evaluate_rays(head, ray_directions, bins, x, region_mask)
                    sigma += head_sigma
                    colors += head_colors

            regenerated_px_values = integrate_ray_color(sigma, delta, colors)

            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            epoch_losses.append(loss.item())
            if iters % 100 == 0:
                print(f'\tIteration: {iters}    Loss: {loss.item():.6f} ')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1

        scheduler.step()

        print(f'Saving checkpoint for epoch {i}.')
        checkpoint_dir = f'./../checkpoints/derf/heads/{DATASET_NAME}/{DATASET_TYPE}'
        checkpoint_path = os.path.join(checkpoint_dir, f'e{i}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save([head.state_dict() for head in model_derf.heads], checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        plt.figure()
        plt.plot(epoch_losses)
        plt.title(f'DeRF reconstruction loss - Epoch {i}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(checkpoint_dir, f'loss_e{i}.png'))
        plt.close()
        print(f'Saved summary visualization for epoch {i}.')


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

    # Train coarse NeRF
    # nerf_train.train(model_nerf_coarse, nerf_optimizer, None, nerf_data_loader,
    #                  near, far, int(DATASET_EPOCHS_COARSE[DATASET_NAME]), NUM_BINS['coarse'], 'derf_coarse')

    # Load coarse NeRF
    model_nerf_coarse.load_state_dict(torch.load(f'./../checkpoints/derf/coarse/{DATASET_NAME}/{DATASET_TYPE}/e0.pt'))

    print('Training Voronoi decomposition')

    model_voronoi = Voronoi(HEAD_COUNT, bounding_box).to(DEVICE)
    voronoi_optimizer = torch.optim.Adam(model_voronoi.parameters(), lr=1e-4)
    voronoi_data_loader = DataLoader(training_dataset, batch_size=16384, shuffle=True)

    # Train soft Voronoi model
    train_voronoi(model_voronoi, voronoi_optimizer, voronoi_data_loader, model_nerf_coarse, near, far)

    # # Load soft Voronoi model
    # model_voronoi.load_state_dict(
    #     torch.load(f'./../checkpoints/derf/voronoi/{DATASET_NAME}/{DATASET_TYPE}/e0.pt')['state_dict'])

    print('Training DeRF with learned head positions')

    model_derf = DeRF(model_voronoi.head_positions)

    derf_data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    derf_optimizer = torch.optim.Adam(model_derf.all_parameters, lr=5e-4 * TRAINING_ACCELERATION)
    derf_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            derf_optimizer, milestones=np.array(DATASET_MILESTONES[DATASET_NAME]) / TRAINING_ACCELERATION, gamma=0.5)

    # # load DeRF checkpoint
    # head_state_dicts = torch.load(f'./../checkpoints/derf/heads/{DATASET_NAME}/{DATASET_TYPE}/e0.pt')
    #
    # for head, state_dict in zip(model_derf.heads, head_state_dicts):
    #     head.load_state_dict(state_dict)

    # Train DeRF
    train_derf(model_derf, model_voronoi, derf_optimizer, derf_scheduler, derf_data_loader,
               near, far, int(DATASET_EPOCHS[DATASET_NAME] / TRAINING_ACCELERATION), NUM_BINS['fine'])



if __name__ == '__main__':
    training_loop()
    print('Training complete, checkpoints and summary visualizations for all epochs saved.')

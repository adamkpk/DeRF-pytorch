import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import (DEVICE,
                    NUM_BINS,
                    DATASET_NAME,
                    DATASET_TYPE,
                    TRAINING_ACCELERATION,
                    DATASET_EPOCHS,
                    DATASET_MILESTONES)

from model.nerf import NeRF, render_rays


def train(model, optimizer, scheduler, data_loader, near, far, epochs, bins, model_type='nerf'):
    iters = 0
    for i in range(epochs):
        print(f'Training NeRF. Epoch: {i}')
        epoch_losses = []  # per-batch losses within epoch
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(DEVICE)
            ray_directions = batch[:, 3:6].to(DEVICE)
            ground_truth_px_values = batch[:, 6:].to(DEVICE)

            regenerated_px_values = render_rays(model, ray_origins, ray_directions, near, far, bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            epoch_losses.append(loss.item())
            if iters % 100 == 0:
                print(f'\tIteration: {iters}    Loss: {loss.item():.6f} ')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1

        if scheduler is not None:
            scheduler.step()

        checkpoint_dir = f'./../checkpoints/{model_type}/{DATASET_NAME}/{DATASET_TYPE}'  # turn back from coarse
        checkpoint_path = os.path.join(checkpoint_dir, f'e{i}.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        # plot and save loss graph for this epoch
        plt.figure()
        plt.plot(epoch_losses)
        plt.title(f'NeRF reconstruction loss - Epoch {i}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(checkpoint_dir, f'loss_epoch_{i}.png'))
        plt.close()
        print(f'Saved summary visualization for epoch {i}.')



def training_loop():
    with open(f'./../data/{DATASET_NAME}_{DATASET_TYPE}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    training_dataset = full_dataset[0]

    near = full_dataset[2]
    far = full_dataset[3]

    model = NeRF(hidden_dim=128).to(DEVICE)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4 * TRAINING_ACCELERATION)
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=np.array(DATASET_MILESTONES[DATASET_NAME]) / TRAINING_ACCELERATION, gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    train(model, model_optimizer, model_scheduler, data_loader,
          near, far, int(DATASET_EPOCHS[DATASET_NAME] / TRAINING_ACCELERATION), NUM_BINS['fine'])


if __name__ == '__main__':
    training_loop()
    print('Training complete, checkpoints and summary visualizations for all epochs saved.')

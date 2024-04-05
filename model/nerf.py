import os
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from config import DATASET_NAME, DATASET_SIZE_DICT, DATASET_TEST_SIZE


class NeRF(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NeRF, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # density estimation
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1))

        # color estimation
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU())
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, dim):
        out = [x]
        for j in range(dim):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        # emb_x: [batch_size, embedding_dim_pos * 6]

        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        # emb_d: [batch_size, embedding_dim_direction * 6]

        h = self.block1(emb_x)
        # h: [batch_size, hidden_dim]

        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        # tmp: [batch_size, hidden_dim + 1]

        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        # h: [batch_size, hidden_dim], sigma: [batch_size]

        h = self.block3(torch.cat((h, emb_d), dim=1))
        # h: [batch_size, hidden_dim // 2]

        c = self.block4(h)
        # c: [batch_size, 3]

        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    dev = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=dev).expand(ray_origins.shape[0], nb_bins)

    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=dev)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=dev).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]

    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background

    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, loader, testing_dataset, dev='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, height=400, width=400):
    training_loss = []

    for i in range(nb_epochs):
        for batch in tqdm(loader):
            ray_origins = batch[:, :3].to(dev)
            ray_directions = batch[:, 3:6].to(dev)
            ground_truth_px_values = batch[:, 6:].to(dev)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

        scheduler.step()

        torch.save(nerf_model.state_dict(), f'./../checkpoints/e{i}.pt')
        print('Saved checkpoint')

        for img_index in range(DATASET_TEST_SIZE[DATASET_NAME]):
            test(nerf_model, hn, hf, testing_dataset, epoch=i, img_index=img_index, nb_bins=nb_bins,
                 height=height, width=width)

    return training_loss


@torch.no_grad()
def test(nerf_model, hn, hf, dataset, chunk_size=10, epoch=0, img_index=0, nb_bins=192, height=400, width=400):
    ray_origins = dataset[img_index * height * width: (img_index + 1) * height * width, :3]
    ray_directions = dataset[img_index * height * width: (img_index + 1) * height * width, 3:6]

    data = []   # list of regenerated pixel values

    for i in tqdm(range(int(np.ceil(height / chunk_size)))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * width * chunk_size: (i + 1) * width * chunk_size].to(device)
        ray_directions_ = ray_directions[i * width * chunk_size: (i + 1) * width * chunk_size].to(device)
        regenerated_px_values = render_rays(nerf_model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)

    img = torch.cat(data).data.cpu().numpy().reshape(height, width, 3)

    img_dir = os.path.join(f'./../results/{DATASET_NAME}', run_date)
    os.makedirs(img_dir, exist_ok=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(img_dir, f'e{epoch}_img{img_index}.png'))


def train_and_test():
    # training_dataset = torch.from_numpy(np.load('./../data/training_data.pkl', allow_pickle=True))
    # testing_dataset = torch.from_numpy(np.load('./../data/testing_data.pkl', allow_pickle=True))

    with open(f'./../data/{DATASET_NAME}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    training_dataset = full_dataset[0]
    testing_dataset = full_dataset[1]

    print('Shapes:', training_dataset.shape, testing_dataset.shape)

    model = NeRF(hidden_dim=256).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model_scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    t_loss = train(model, model_optimizer, model_scheduler, data_loader, testing_dataset,
                   nb_epochs=5, dev=device, hn=2, hf=6, nb_bins=192,
                   height=DATASET_SIZE_DICT[DATASET_NAME][1], width=DATASET_SIZE_DICT[DATASET_NAME][0])

    print('Training loss:', t_loss)


def test_last_epoch():
    with open(f'./../data/{DATASET_NAME}_data.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    testing_dataset = full_dataset[1]

    # testing_dataset = torch.from_numpy(np.load('./../data/testing_data.pkl', allow_pickle=True))

    epoch = 4

    model = NeRF(hidden_dim=256).to(device)

    model.load_state_dict(torch.load(f'./../checkpoints/{DATASET_NAME}/e{epoch}.pt'))
    # model.eval()

    for img_index in range(DATASET_TEST_SIZE[DATASET_NAME]):
        test(model, hn=2, hf=6, dataset=testing_dataset, epoch=epoch, img_index=img_index, nb_bins=192,
             height=DATASET_SIZE_DICT[DATASET_NAME][1], width=DATASET_SIZE_DICT[DATASET_NAME][0])


device = 'cuda'

run_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# train_and_test()

test_last_epoch()

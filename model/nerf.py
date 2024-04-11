import torch
import torch.nn as nn

from config import (DEVICE,
                    HIDDEN_UNITS,
                    DATASET_NAME,
                    DATASET_WHITEBG_EQUALIZATION)


class NeRF(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=HIDDEN_UNITS['full']):
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

        radiance = self.block4(h)
        # c: [batch_size, 3]

        # radiance, density
        return radiance, sigma


def sample_ray_positions(ray_origins, ray_directions, near, far, bins):
    t = torch.linspace(near, far, bins, device=DEVICE).expand(ray_origins.shape[0], bins)

    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=DEVICE)
    t = lower + (upper - lower) * u  # [batch_size, bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1],
                       torch.tensor([1e10], device=DEVICE).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, bins, 3]

    return x, delta


def evaluate_rays(model, ray_directions, bins, x, mask=None):
    # Expand the ray_directions tensor to match the shape of x: [batch_size, bins, 3]
    ray_directions = ray_directions.expand(bins, ray_directions.shape[0], 3).transpose(0, 1)

    if mask is not None:
        # Only feed entries that pass through the mask to the model
        masked_colors, masked_sigma = model(x[mask == 1], ray_directions[mask == 1])

        # Fill dropped components with 0
        colors = torch.zeros((ray_directions.shape[0], bins, 3)).to(DEVICE)
        colors[mask == 1] = masked_colors
        sigma = torch.zeros((ray_directions.shape[0], bins)).to(DEVICE)
        sigma[mask == 1] = masked_sigma

    else:
        colors, sigma = model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
        colors = colors.reshape(x.shape)
        sigma = sigma.reshape(x.shape[:-1])

    return sigma, colors


def integrate_ray_color(sigma, delta, colors):
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, bins]

    transmittance = torch.cumprod(1 - alpha, 1)
    transmittance = torch.cat((torch.ones((transmittance.shape[0], 1), device=DEVICE),
                               transmittance[:, :-1]), dim=-1)

    weights = transmittance.unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)

    if DATASET_WHITEBG_EQUALIZATION[DATASET_NAME]:
        weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
        return c + 1 - weight_sum.unsqueeze(-1)

    return c


def render_rays(model, ray_origins, ray_directions, near, far, bins):
    x, delta = sample_ray_positions(ray_origins, ray_directions, near, far, bins)
    sigma, colors = evaluate_rays(model, ray_directions, bins, x)

    return integrate_ray_color(sigma, delta, colors)

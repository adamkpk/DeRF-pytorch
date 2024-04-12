import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from config import (DEVICE,
                    HIDDEN_UNITS)

from nerf import NeRF


class Voronoi(nn.Module):
    def __init__(self, head_count, bounding_box):
        super(Voronoi, self).__init__()

        self.head_positions = nn.Parameter(torch.rand(head_count, 3)
                                           * (bounding_box[1] - bounding_box[0]) + bounding_box[0])

        self.softmax_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, positions):
        d = torch.norm(positions.unsqueeze(-2) - self.head_positions, dim=-1)
        return nn.functional.softmax(-self.softmax_scale * d, dim=-1)


class DeRF:
    def __init__(self, head_positions):
        self.head_positions = head_positions

        self.heads = nn.ModuleList()

        for _ in range(len(head_positions)):
            model = NeRF(hidden_dim=HIDDEN_UNITS['head']).to(DEVICE)
            self.heads.append(model)

        # Aggregate all parameters from all heads
        self.all_parameters = [param for head in self.heads for param in head.parameters()]


def ray_contributions(sigma, delta, voronoi_weights):
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, bins]

    transmittance = torch.cumprod(1 - alpha, 1)
    transmittance = torch.cat((torch.ones((transmittance.shape[0], 1), device=DEVICE),
                               transmittance[:, :-1]), dim=-1)

    weights = transmittance.unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a sum of densities along each ray
    v = (weights * voronoi_weights).sum(dim=1)

    return v


# Takes a batch of [batch_size, bins, 3] ray samples, returns the nearest Voronoi patch indexes to each sample per ray
# This KDTree implementation is much shorter than the Delaunay implementation, but is theoretically slower
def partition_samples(batch, cell_origins):
    kd = cKDTree(cell_origins)

    _, nearests = kd.query(batch, k=1)

    return nearests




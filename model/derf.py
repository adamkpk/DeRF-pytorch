import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from config import (DEVICE,
                    HIDDEN_UNITS)

from nerf import NeRF


class Voronoi(nn.Module):
    def __init__(self, head_count, bounding_box):
        super(Voronoi, self).__init__()

        self.head_count = head_count
        self.bounding_box = bounding_box

        # self.head_positions = nn.Parameter(torch.rand(head_count, 3)
        #                                    * (bounding_box[1] - bounding_box[0]) + bounding_box[0])

        # Voronoi initialization scheme: split bounding box into 8 sub-boxes (octree-like) then uniformly distribute.
        def patchwise_uniform():
            min_corner, max_corner = self.bounding_box

            mid = (max_corner + min_corner) / 2
            mid_x, mid_y, mid_z = mid

            boxes = [
                (min_corner, mid),
                ((mid_x, min_corner[1], min_corner[2]), (max_corner[0], mid_y, mid_z)),
                ((min_corner[0], mid_y, min_corner[2]), (mid_x, max_corner[1], mid_z)),
                ((mid_x, mid_y, min_corner[2]), (max_corner[0], max_corner[1], mid_z)),
                ((min_corner[0], min_corner[1], mid_z), (mid_x, mid_y, max_corner[2])),
                ((mid_x, min_corner[1], mid_z), (max_corner[0], mid_y, max_corner[2])),
                ((min_corner[0], mid_y, mid_z), (mid_x, max_corner[1], max_corner[2])),
                (mid, max_corner)
            ]

            heads_per_box = self.head_count // len(boxes)

            box_heads = torch.empty((self.head_count, 3))

            # for i, box in enumerate(boxes):
            #     box_heads[i * heads_per_box:(i + 1) * heads_per_box]\
            #         = (torch.rand(heads_per_box, 3) * (torch.Tensor(box[1]) - torch.Tensor(box[0]))
            #            + torch.Tensor(box[0]))

            for i, box in enumerate(boxes):
                box_heads[i * heads_per_box:(i + 1) * heads_per_box]\
                    = ((torch.Tensor(box[1]) + torch.Tensor(box[0])) / 2)

            return box_heads

        # def perturbed_lattice():
        #     min_corner, max_corner = self.bounding_box
        #
        #     step = (max_corner - min_corner) / (self.head_count ** (1 / 3))
        #
        #     points = []
        #     for x in torch.arange(min_corner[0], max_corner[0], step[0]):
        #         for y in torch.arange(min_corner[1], max_corner[1], step[1]):
        #             for z in torch.arange(min_corner[2], max_corner[2], step[2]):
        #                 points.append(torch.Tensor([x, y, z]) + 0.5 * step * (torch.rand(3) - 0.5))
        #
        #     return torch.Tensor(points)

        self.head_positions = nn.Parameter(patchwise_uniform().to(DEVICE))

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

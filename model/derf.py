import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from config import (DEVICE,
                    HIDDEN_UNITS,
                    VORONOI_INIT_SCHEME)

from nerf import NeRF


class Voronoi(nn.Module):
    def __init__(self, head_count, bounding_box):
        super(Voronoi, self).__init__()

        self.head_count = head_count
        self.bounding_box = bounding_box

        """
        init_heads(scheme): Selects initialization scheme for Voronoi heads.
            'uniform': random uniform across the scene bounding box.
            'stratified_uniform': splits bounding box into 8 sub-cubes, initializes (n/8) heads uniformly at random in
                each sub-cube (assumption: head_count is a power of 2 and >= 8)
            'determinstic_grid': splits bounding box into 8 sub-cubes, inits 1 head at the center of each box
        """
        def init_heads(scheme=VORONOI_INIT_SCHEME):
            min_corner, max_corner = self.bounding_box
            head_pos = torch.empty((self.head_count, 3))  # head positions

            if scheme == 'uniform':
                head_pos = torch.rand(head_count, 3) * (max_corner - min_corner) + min_corner

            else:
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
                if scheme == 'stratified_uniform':
                    assert self.head_count >= 8 and (self.head_count & (self.head_count - 1)) == 0, \
                        "'stratified_uniform' init scheme requires n >= 8 heads, where n is a power of 2"

                    for i, box in enumerate(boxes):
                        head_pos[i * heads_per_box:(i + 1) * heads_per_box]\
                            = (torch.rand(heads_per_box, 3) * (torch.Tensor(box[1]) - torch.Tensor(box[0]))
                               + torch.Tensor(box[0]))

                elif scheme == 'deterministic_grid':
                    assert self.head_count == 8, \
                        "'deterministic_grid' init scheme can only be used for head counts with integer cube roots."

                    for i, box in enumerate(boxes):
                        head_pos[i * heads_per_box:(i + 1) * heads_per_box]\
                            = ((torch.Tensor(box[1]) + torch.Tensor(box[0])) / 2)

            return head_pos

        self.head_positions = nn.Parameter(init_heads().to(DEVICE))

        # 'temperature' parameter; see paper section 3.3. Deprecated (see report discussion).
        self.softmax_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, positions):
        d = torch.norm(positions.unsqueeze(-2) - self.head_positions, dim=-1)
        # softmin (negative softmax) on distances to head centers
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


# Calculates the region density weighted contributions of each decomposition cell integrated across a batch of rays
def ray_contributions(sigma, delta, voronoi_weights):
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, bins]

    transmittance = torch.cumprod(1 - alpha, 1)
    transmittance = torch.cat((torch.ones((transmittance.shape[0], 1), device=DEVICE),
                               transmittance[:, :-1]), dim=-1)

    weights = transmittance.unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a sum of densities along each ray
    v = (weights * voronoi_weights).sum(dim=1)

    return v


# partitions spatial ray samples into Voronoi regions given cell centers
# deprecated/ for A/B testing ; we calculate this via Voronoi model forward pass + argmax
def partition_samples(batch, cell_origins):
    kd = cKDTree(cell_origins)

    _, nearests = kd.query(batch, k=1)

    return nearests

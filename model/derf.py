import torch
import torch.nn as nn

from config import DEVICE

from nerf import NeRF


class Voronoi(nn.Module):
    def __init__(self, head_count, bounding_box):
        super(Voronoi, self).__init__()

        self.heads = nn.Parameter(torch.rand(head_count, 3)
                                  * (bounding_box[1] - bounding_box[0]) + bounding_box[0])

        self.softmax_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, positions):
        d = torch.norm(positions.unsqueeze(-2) - self.heads, dim=-1)
        return nn.functional.softmax(-self.softmax_scale * d, dim=-1)


class DeRF(nn.Module):
    def __init__(self, heads, ):
        super(DeRF, self).__init__()

    def forward(self):
        pass


def ray_contributions(sigma, delta, voronoi_weights):
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, bins]

    transmittance = torch.cumprod(1 - alpha, 1)
    transmittance = torch.cat((torch.ones((transmittance.shape[0], 1), device=DEVICE),
                               transmittance[:, :-1]), dim=-1)

    weights = transmittance.unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a sum of densities along each ray
    v = (weights * voronoi_weights).sum(dim=1)

    return v

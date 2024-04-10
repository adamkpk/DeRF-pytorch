import torch
import torch.nn as nn

from config import DEVICE

from nerf import NeRF


class Voronoi(nn.Module):
    def __init__(self, heads, bounding_box):
        super(Voronoi, self).__init__()

        self.heads = heads
        self.head_centers = nn.Parameter(torch.rand(self.heads, 3)
                                         * (bounding_box[1] - bounding_box[0]) + bounding_box[0])

        self.softmax_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, positions):
        d = torch.norm(positions.unsqueeze(-2) - self.head_centers / self.center_scale, dim=-1)
        return nn.functional.softmax(-self.softmax_scale * d, dim=-1)


class DeRF(nn.Module):
    pass

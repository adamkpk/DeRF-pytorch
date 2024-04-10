import torch
import torch.nn as nn

from config import DEVICE

from nerf import NeRF


class Voronoi(nn.Module):
    def __init__(self, heads):
        super(Voronoi, self).__init__()

        self.heads = heads

    def forward(self):
        pass


class DeRF(nn.Module):
    pass

import torch
import math
import numpy as np

from diffoptics.distributions.BaseDistribution import BaseDistribution


class GaussianDistribution(BaseDistribution):

    def __init__(self, mean=0.0, std=1.0, eps=1e-15):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def sample(self, nb_points, device='cpu'):
        return self.mean + self.std * torch.randn(nb_points, device=device)

    def pdf(self, x):
        return 1 / (self.std * np.sqrt(2 * math.pi)) * torch.exp((- 1 / 2) * ((x - self.mean) / self.std) ** 2)

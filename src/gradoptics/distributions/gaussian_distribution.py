import torch
import math
import numpy as np

from gradoptics.distributions.base_distribution import BaseDistribution


class GaussianDistribution(BaseDistribution):
    """
    1D Gaussian Distribution.
    """

    def __init__(self, mean=0.0, std=1.0):
        """
        :param mean: mean of the distribution (:obj:`float`)
        :param std: standard deviation of the distribution (:obj:`float`)
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def sample(self, nb_points, device='cpu'):
        return self.mean + self.std * torch.randn(nb_points, device=device)

    def pdf(self, x):
        """
        Returns the pdf function evaluated at ``x``

        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)

        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        return 1 / (self.std * np.sqrt(2 * math.pi)) * torch.exp((- 1 / 2) * ((x - self.mean) / self.std) ** 2)

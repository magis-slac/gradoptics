import torch
import math

from gradoptics.distributions.base_distribution import BaseDistribution



class PointSource(BaseDistribution):
    """
    Point source radiating uniformly in 4pi.
    """

    def __init__(self, n=int(1e6), position=[0.31, 0., 0.]):
        """
        :param n: Number of atoms (:obj:`int`)
        :param position: Position of the center of the atom cloud [m] (:obj:`list`)
        """
        super().__init__()
        self.n = n
        self.position = torch.tensor(position)

    def sample(self, nb_points, device='cpu'):
        return self.position.repeat(nb_points, 1)

    def pdf(self, x):
        raise NotImplemented

    def plot(self, ax, **kwargs):
        """
        Plots the center of the atom cloud on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        ax.scatter(self.position[0], self.position[1], self.position[2], **kwargs)

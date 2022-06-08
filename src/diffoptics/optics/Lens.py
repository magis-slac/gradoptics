import abc
import torch
import numpy as np

from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.optics.Ray import Rays
from diffoptics.optics.Vector import batch_vector
from diffoptics.transforms.SimpleTransform import SimpleTransform


class Lens(BaseOptics, abc.ABC):
    """
    Base class for lenses.
    """

    @abc.abstractmethod
    def sample_points_on_lens(self, nb_points, device='cpu'):
        """
        Sample points uniformly on the lens

        :param nb_points: Number of points to sample (:obj:`int`)
        :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

        :return: Sampled points (:obj:`torch.tensor`)
        """
        raise NotImplemented


class PerfectLens(Lens):
    """
    Models a thin lens.
    """

    def __init__(self, f=0.05, na=0.714, position=[0., 0., 0.], m=0.15, transform=None, eps=1e-15):
        """
        :param f: Focal length (:obj:`float`)
        :param na: Inverse of the f-number (:obj:`float`)
        :param position: Position of the lens (:obj:`list`)
        :param m: Lens magnification  (:obj:`float`)
        :param transform: Transform to orient the lens (:py:class:`~diffoptics.transforms.BaseTransform.BaseTransform`)
        :param eps: Parameter used for numerical stability in the different class methods (:obj:`float`). Default
                    is ``'1e-15'``
        """
        super(PerfectLens, self).__init__()
        self.f = f
        self.na = na
        self.m = m
        self.pof = f * (m + 1) / m
        self.eps = eps
        if transform is None:
            self.transform = SimpleTransform(0., 0., 0., torch.tensor(position))
        else:
            self.transform = transform

    def get_ray_intersection(self, incident_rays):
        incident_rays = self.transform.apply_inverse_transform(incident_rays)  # World space to lens space
        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (- origins[:, 0]) / (directions[:, 0] + self.eps)
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]

        # Check that the intersection is real (t>0) and within the lens' bounds
        condition = (t > 0) & ((y ** 2 + z ** 2) < (self.f * self.na / 2) ** 2)
        t[~condition] = float('nan')
        return t

    def intersect(self, incident_rays, t):
        incident_rays = self.transform.apply_inverse_transform(incident_rays)  # World space to lens space

        x_objects = incident_rays.origins
        d_0 = incident_rays.directions

        # If x_objects is on the left of the lens, put the plane of focus on the left as well
        pof = self.pof * torch.sign(x_objects[:, 0])
        # Put the camera on the opposite side of the lens with respect to x_objects
        camera_pos = - self.f * (1 + self.m) * torch.sign(x_objects[:, 0])

        # Intersections with the lens
        x_0 = x_objects + t.unsqueeze(1) * d_0

        # computes x_star, the intersections of the rays with the plane of focus
        t = (x_0[:, 0] - pof) / (d_0[:, 0] + self.eps)
        t = -t
        x_star = x_0 + t.unsqueeze(1) * d_0

        # Computes x_v, the intersections of the rays coming from x_star and passing trough the optical center with the
        # camera
        d = - x_star
        t = (camera_pos - x_star[:, 0]) / (d[:, 0] + self.eps)
        x_v = x_star + t.unsqueeze(1) * d
        _d_out = batch_vector(x_v[:, 0] - x_0[:, 0], x_v[:, 1] - x_0[:, 1], x_v[:, 2] - x_0[:, 2])

        return self.transform.apply_transform(Rays(x_0, _d_out, luminosities=incident_rays.luminosities,
                                                   meta=incident_rays.meta, device=incident_rays.device))

    def sample_points_on_lens(self, nb_points, device='cpu'):

        points = torch.zeros((nb_points, 3), device=device)
        lens_radius = self.f * self.na / 2
        # Sampling uniformly on a disk.
        # See https://stats.stackexchange.com/questions/481543/generating-random-points-uniformly-on-a-disk
        r_squared = torch.rand(nb_points, device=device) * lens_radius ** 2
        theta = torch.rand(nb_points, device=device) * 2 * np.pi
        points[:, 1] = torch.sqrt(r_squared) * torch.cos(theta)
        points[:, 2] = torch.sqrt(r_squared) * torch.sin(theta)
        return self.transform.apply_transform_(points)

    def plot(self, ax, s=0.1, color='lightblue', resolution=100):
        # @Todo, change this to plot_surface
        y = torch.arange(-self.f * self.na / 2, self.f * self.na / 2, (self.f * self.na) / resolution)
        z = torch.arange(-self.f * self.na / 2, self.f * self.na / 2, (self.f * self.na) / resolution)
        y, z = torch.meshgrid(y, z)

        y = y.reshape(-1)
        z = z.reshape(-1)
        x = torch.zeros(resolution * resolution)

        indices = (y ** 2 + z ** 2) < (self.f * self.na / 2) ** 2
        x = x[indices]
        y = y[indices]
        z = z[indices]

        # Lens space to world space
        xyz = self.transform.apply_transform_(torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1))

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, c=color)


class ThickLens(Lens):
    """
    Models a thick lens using Snell's law.
    """

    def __init__(self):
        super(ThickLens, self).__init__()
        pass

    @torch.no_grad()
    def get_ray_intersection(self, incident_rays):
        raise NotImplemented

    def intersect(self, incident_rays, t):
        raise NotImplemented

    def sample_points_on_lens(self, nb_points, device='cpu', eps=1e-15):
        raise NotImplemented

    def plot(self, ax):
        raise NotImplemented

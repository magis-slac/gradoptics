import math
import torch

from gradoptics import Rays
from gradoptics.light_sources.base_light_source import BaseLightSource
from gradoptics.optics import batch_vector, normalize_batch_vector


class LightSourceFromDistribution(BaseLightSource):
    """
    Models a light source from a distribution. It emits rays in 4pi, with their origins sampled from the distribution.
    """

    def __init__(self, distribution, bounding_shape=None):
        """
        :param distribution: Distribution from which photons will be sampled
                             (:py:class:`~gradoptics.distributions.base_distribution.BaseDistribution`)
        :param bounding_shape: A bounding shape that bounds the light source
                               (:py:class:`~gradoptics.optics.bounding_shape.BoundingShape`). Default is ``None``

        .. note::
             A bounding shape is required if this light source is used with backward ray tracing

        """
        self.distribution = distribution
        self.bounding_shape = bounding_shape

    def sample_rays(self, nb_rays, device='cpu', sample_in_2pi=False):

        # Sample rays in 4 pi
        azimuthal_angle = torch.rand(nb_rays) * 2 * math.pi
        polar_angle = torch.arccos(1 - 2 * torch.rand(nb_rays))

        emitted_direction = batch_vector(torch.sin(polar_angle) * torch.sin(azimuthal_angle),
                                         torch.sin(polar_angle) * torch.cos(azimuthal_angle),
                                         torch.cos(polar_angle))
        # Sample in 2 pi
        if sample_in_2pi:
            emitted_direction[:, 0] = -emitted_direction[:, 0].abs()

        del azimuthal_angle
        del polar_angle
        torch.cuda.empty_cache()

        return Rays(self.distribution.sample(nb_rays, device=device).type(emitted_direction.dtype), emitted_direction,
                    device=device)

    def get_pointed_rays(self, nb_rays, target_point, device='cpu'):
        # Sample origins from underlying distribution
        origins = self.distribution.sample(nb_rays, device=device)
        directions = normalize_batch_vector(target_point - origins)

        # TODO: implement intensity weights based on solid angle calculation
        # Solid angle calculation reference: http://websites.umich.edu/~ners311/CourseLibrary/SolidAngleOfADiskOffAxis.pdf
        weights = torch.ones(origins.shape[0])

        torch.cuda.empty_cache()

        return Rays(origins, directions, luminosities=weights,
                    device=device)

    def plot(self, ax, **kwargs):
        return self.distribution.plot(ax, **kwargs)

    def pdf(self, x):
        """
        Returns the pdf function of the distribution evaluated at ``x``

        .. warning::
           The pdf may be unnormalized

        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)

        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        return self.distribution.pdf(x)

import math
import torch

from diffoptics import Rays
from diffoptics.light_sources.BaseLightSource import BaseLightSource
from diffoptics.optics import batch_vector


class LightSourceFromDistribution(BaseLightSource):

    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

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

        return Rays(self.distribution.sample(nb_rays, device=device).type(emitted_direction.dtype), emitted_direction, device=device)

    def plot(self, ax, **kwargs):
        return self.distribution.plot(ax, **kwargs)

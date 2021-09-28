import torch

from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.optics.Vector import normalize_vector, normalize_batch_vector


class Ray(BaseOptics):

    def __init__(self, origin, direction, luminosity=torch.tensor([1.]), meta={}, device='cpu'):
        super(Ray, self).__init__()
        self.origin = origin.to(device)
        self.direction = normalize_vector(direction).to(device)
        self.luminosity = luminosity.to(device)
        self.meta = meta
        self.device = device

    def plot(self, ax):
        # @Todo
        raise NotImplementedError("Not implemented yet.")


class Rays:

    def __init__(self, origins, directions, luminosities=None, meta={}, device='cpu'):
        self.origins = origins.to(device)
        self.directions = normalize_batch_vector(directions).to(device)
        self.luminosities = luminosities.to(device) if luminosities is not None else None
        self.meta = meta
        self.device = device

        assert origins.dtype == directions.dtype
        # Sanity check: dimensionality consistency
        assert origins.shape == directions.shape
        if luminosities is not None:
            assert origins.shape[0] == luminosities.shape[0]
        for key in meta.keys():
            assert meta[key].shape[0] == origins.shape[0]

    def get_at(self, condition):
        meta = {}
        for key in self.meta.keys():
            meta[key] = self.meta[key][condition]

        return Rays(self.origins[condition],
                    self.directions[condition],
                    luminosities=self.luminosities[condition] if self.luminosities is not None else None,
                    meta=meta,
                    device=self.device)

    def update_at(self):
        # @Todo & o & d private!
        raise NotImplementedError("Not implemented yet.")

    def get_origin_and_direction(self):
        # @Todo & o & d private!
        raise NotImplementedError("Not implemented yet.")

    def get_size(self):
        return self.origins.shape[0]

    def plot(self, ax, t, color='C0', line_width=None):
        # @ Todo, parallelize
        for i in range(len(t)):
            point = self.origins[i] + t[i] * self.directions[i]

            ax.plot([self.origins[i, 0], point[0]],
                    [self.origins[i, 1], point[1]],
                    [self.origins[i, 2], point[2]],
                    color=color, linewidth=line_width)
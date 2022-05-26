import torch

from diffoptics.optics.Vector import normalize_vector, normalize_batch_vector


class Ray:
    """
    Models a ray characterized by an origin and a direction. The ray carries some luminosity and meta data which can
    be used for storing information, for example during surface intersections.
    """

    def __init__(self, origin, direction, luminosity=torch.tensor([1.]), meta={}, device='cpu'):
        """
        :param origin:
        :param direction:
        :param luminosity:
        :param meta:
        :param device:
        """

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
    """
    Models a batch of rays characterized by their origins and directions. The rays carry some luminosity and meta data
    which can be used for storing information, for example during surface intersections.
    """

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

    def __getitem__(self, condition):
        meta = {}
        for key in self.meta.keys():
            meta[key] = self.meta[key][condition]

        return Rays(self.origins[condition], self.directions[condition],
                    luminosities=self.luminosities[condition] if self.luminosities is not None else None,
                    meta=meta, device=self.device)

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


def empty_like(rays: Rays):
    origins = torch.empty_like(rays.origins)
    directions = torch.empty_like(rays.directions)
    luminosities = (torch.empty_like(rays.luminosities)) if rays.luminosities is not None else None
    meta = {}
    for key in rays.meta.keys():
        meta[key] = torch.empty_like(rays.meta[key])

    return Rays(origins, directions, luminosities=luminosities, device=rays.device, meta=meta)


def cat(rays1: Rays, rays2: Rays):
    rays1.origins = torch.cat((rays1.origins, rays2.origins))
    rays1.directions = torch.cat((rays1.directions, rays2.directions))

    if rays1.luminosities is not None:
        rays1.luminosities = torch.cat((rays1.luminosities, rays2.luminosities))

    for key in rays1.meta.keys():
        rays1[key] = torch.cat((rays1.meta[key], rays2.meta[key]), dim=0)

    return rays1

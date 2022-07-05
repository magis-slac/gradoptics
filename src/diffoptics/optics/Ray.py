import torch

from diffoptics.optics.Vector import normalize_vector, normalize_batch_vector


class Ray:
    """
    Models a ray characterized by an origin and a direction. The ray carries some luminosity and meta data which can
    be used for storing information, for example during surface intersections.
    """

    def __init__(self, origin, direction, luminosity=torch.tensor([1.]), meta=None, device='cpu'):
        """
        :param origin: Origin of the ray (:obj:`torch.tensor`)
        :param direction: Direction of the ray (:obj:`torch.tensor`)
        :param luminosity: Luminosity carried by the ray (:obj:`torch.tensor`)
        :param meta: Meta data carried by the ray (:obj:`dict`)
        :param device: The desired device of returned tensor (:obj:`str`)
        """

        super(Ray, self).__init__()

        self.origin = origin.to(device)
        self.direction = normalize_vector(direction).to(device)
        self.luminosity = luminosity.to(device)
        self.meta = meta if meta is not None else {}
        self.device = device

    def plot(self, ax, t):
        """
        Plots the ray at time ``t``

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        :param t: times t (:obj:`torch.tensor`)
        """
        raise NotImplementedError("Not implemented yet.")


class Rays:
    """
    Models a batch of rays characterized by their origins and directions. The rays carry some luminosity and meta data
    which can be used for storing information, for example during surface intersections.
    """

    def __init__(self, origins, directions, luminosities=None, meta=None, device='cpu'):
        """
        :param origins: Origins of the rays (:obj:`torch.tensor`)
        :param directions: Directions of the rays (:obj:`torch.tensor`)
        :param luminosities: Luminosities carried by the rays (:obj:`torch.tensor`)
        :param meta: Meta data carried by the rays (:obj:`dict`)
        :param device: The desired device of returned tensor (:obj:`str`)
        """
        self.origins = origins.to(device)
        self.directions = normalize_batch_vector(directions).to(device)
        self.luminosities = luminosities.to(device) if luminosities is not None else torch.ones(origins.shape[0],
                                                                                                device=device)
        self.meta = meta if meta is not None else {}
        self.device = device

        assert self.origins.dtype == self.directions.dtype
        # Sanity check: dimensionality consistency
        assert self.origins.shape == self.directions.shape
        assert self.origins.shape[0] == self.luminosities.shape[0]
        for key in self.meta.keys():
            assert self.meta[key].shape[0] == self.origins.shape[0]

    def __getitem__(self, condition):
        """
        Returns the rays where the ``condition`` is true

        :param condition: Boolean tensor (:obj:`torch.tensor`)

        :return: A batch of rays where the ``condition`` is true (:py:class:`~diffoptics.optics.Ray.Rays`)
        """
        meta = {}
        for key in self.meta.keys():
            meta[key] = self.meta[key][condition]

        return Rays(self.origins[condition], self.directions[condition], luminosities=self.luminosities[condition],
                    meta=meta, device=self.device)

    def __setitem__(self, condition, value):
        """
        Update the rays where the ``condition`` is true

        :param condition: Boolean tensor (:obj:`torch.tensor`)
        :param value: The new rays (:py:class:`~diffoptics.optics.Ray.Rays`)
        """

        self.origins[condition] = value.origins
        self.directions[condition] = value.directions
        self.luminosities[condition] = value.luminosities

        assert len(self.meta.keys()) == len(value.meta.keys())

        for key in self.meta.keys():
            self.meta[key][condition] = value.meta[key]

    def __call__(self, t):
        """
        Returns the positions of the rays at times ``t``

        :param t: Times t (:obj:`torch.tensor`)

        :return: The positions of the rays at times t (:obj:`torch.tensor`)
        """
        return self.origins + t.unsqueeze(1) * self.directions

    def get_origin_and_direction(self):
        """
        Returns the origins and directions of the rays

        :return: The origins and directions of the batch of rays (:obj:`tuple`)
        """
        return self.origins, self.directions

    def get_size(self):
        """
        Returns the number of rays in the batch

        :return: The number of rays in the batch (:obj:`int`)
        """
        return self.origins.shape[0]

    def plot(self, ax, t, **kwargs):
        """
        Plots the rays at times ``t``

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        :param t: times t (:obj:`torch.tensor`)
        """
        for i in range(len(t)):  # @Todo, parallelize
            point = self.origins[i] + t[i] * self.directions[i]

            ax.plot([self.origins[i, 0], point[0]], [self.origins[i, 1], point[1]], [self.origins[i, 2], point[2]],
                    **kwargs)


def empty_like(rays):
    """
    Returns an uninitialized batch of rays with the same size as ``rays``

    :param rays: the size of ``rays`` will determine the size of the new batch of rays
                 (:py:class:`~diffoptics.optics.Ray.Rays`)

    :return: A new batch of rays (:py:class:`~diffoptics.optics.Ray.Rays`)
    """
    origins = torch.empty_like(rays.origins)
    directions = torch.empty_like(rays.directions)
    luminosities = torch.empty_like(rays.luminosities)
    meta = {}
    for key in rays.meta.keys():
        meta[key] = torch.empty_like(rays.meta[key])

    return Rays(origins, directions, luminosities=luminosities, device=rays.device, meta=meta)


def cat(rays1, rays2):
    """
    Concatenates the two batch of rays

    :param rays1: First batch of rays (:py:class:`~diffoptics.optics.Ray.Rays`)
    :param rays2: Second batch of rays (:py:class:`~diffoptics.optics.Ray.Rays`)

    :return New batch of rays (:py:class:`~diffoptics.optics.Ray.Rays`)
    """
    rays1.origins = torch.cat((rays1.origins, rays2.origins))
    rays1.directions = torch.cat((rays1.directions, rays2.directions))
    rays1.luminosities = torch.cat((rays1.luminosities, rays2.luminosities))

    for key in rays1.meta.keys():
        rays1.meta[key] = torch.cat((rays1.meta[key], rays2.meta[key]), dim=0)

    return rays1

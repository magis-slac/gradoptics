import abc  # Abstract Base Classes


class BaseLightSource(abc.ABC):
    """
    Base class for light sources.
    """

    @abc.abstractmethod
    def sample_rays(self, nb_rays, device='cpu'):
        """
        Samples rays from the light source

        :param nb_rays: Number of rays to sample (:obj:`int`)
        :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

        :return: Sampled rays (:py:class:`~diffoptics.optics.Ray.Rays`)
        """
        return NotImplemented

    @abc.abstractmethod
    def plot(self, ax):
        """
        Plots the position of the light source on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        return NotImplemented

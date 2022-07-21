import abc  # Abstract Base Classes
import torch


class BaseOptics(abc.ABC):
    """
    Base class for optical elements.
    """

    @abc.abstractmethod
    def get_ray_intersection(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the object.

        :param incident_rays: The incident rays (:py:class:`~gradoptics.optics.Ray.Rays`)

        :return: Times t (:obj:`torch.tensor`)
        """
        return NotImplemented

    @abc.abstractmethod
    @torch.no_grad()
    def intersect(self, incident_rays, t):
        """
        Returns the rays reflected or refracted by the optical element.

        :param incident_rays: The incident rays (:py:class:`~gradoptics.optics.Ray.Rays`)
        :param t: The times at which the incident rays will intersect the optical element (:obj:`torch.tensor`)

        :return: (:obj:`tuple`)
                 - Reflected or refracted rays (:py:class:`~gradoptics.optics.Ray.Rays`)
                 - A boolean mask that indicates which incident rays were reflected or refracted ― some rays may not be
                   reflected or refracted for example, due to total internal reflection, which is not modelled at this
                   point (:obj:`torch.tensor`)
        """
        return NotImplemented

    @abc.abstractmethod
    def plot(self, ax):
        """
        Plots the optical element on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        return NotImplemented

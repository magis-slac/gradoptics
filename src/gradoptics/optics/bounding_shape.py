import abc  # Abstract Base Classes
import torch

from gradoptics.optics import BaseOptics


class BoundingShape(BaseOptics):
    """
    Base class for bounding shapes.
    """

    @abc.abstractmethod
    def get_ray_intersection_(self, incident_rays):
        """
        Computes the times t_min and t_max at which the incident rays will intersect the bounding shape

        :param incident_rays: The incident rays (:py:class:`~gradoptics.optics.ray.Rays`)

        :return: (:obj:`tuple`)
                 - Times t_min (:obj:`torch.tensor`)
                 - Times t_max (:obj:`torch.tensor`)
        """
        return NotImplemented
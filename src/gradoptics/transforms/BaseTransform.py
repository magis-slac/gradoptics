import abc  # Abstract Base Classes
import torch
from gradoptics.optics.Ray import Rays


class BaseTransform(abc.ABC):
    """
    Base class for transforms. Enables to switch between world-space and object-space.
    """

    def apply_transform(self, rays):
        """
        Transform the coordinates of ``rays`` in world-space to coordinates in object-space

        :param rays: Rays in world-space (:py:class:`~gradoptics.optics.Ray.Rays`)

        :return: New rays in object-space (:py:class:`~gradoptics.optics.Ray.Rays`)
        """
        new_o = torch.matmul(self.transform.to(rays.device),
                             torch.cat((rays.origins.type(torch.double), torch.ones((rays.origins.shape[0], 1),
                                                                                    device=rays.device)),
                                       dim=1).unsqueeze(-1))[:, :3, 0].type(rays.origins.dtype)
        new_d = torch.matmul(self.transform.to(rays.device),
                             torch.cat((rays.directions.type(torch.double),
                                        torch.zeros((rays.directions.shape[0], 1), dtype=torch.double,
                                                    device=rays.device)), dim=1).unsqueeze(-1))[:, :3, 0].type(
            rays.directions.dtype)
        return Rays(new_o, new_d, luminosities=rays.luminosities, meta=rays.meta, device=rays.device)

    def apply_inverse_transform(self, rays):
        """
        Transform the coordinates of ``rays`` in object-space to coordinates in world-space

        :param rays: Rays in object-space (:py:class:`~gradoptics.optics.Ray.Rays`)

        :return: New rays in world-space (:py:class:`~gradoptics.optics.Ray.Rays`)
        """
        new_o = torch.matmul(self.inverse_transform.to(rays.device),
                             torch.cat((rays.origins.type(torch.double),
                                        torch.ones((rays.origins.shape[0], 1), dtype=torch.double,
                                                   device=rays.device)), dim=1).unsqueeze(-1))[:, :3, 0].type(
            rays.origins.dtype)
        new_d = torch.matmul(self.inverse_transform.to(rays.device),
                             torch.cat((rays.directions.type(torch.double), torch.zeros((rays.directions.shape[0], 1),
                                                                                        device=rays.device)),
                                       dim=1).unsqueeze(-1))[:, :3, 0].type(rays.directions.dtype)
        return Rays(new_o, new_d, luminosities=rays.luminosities, meta=rays.meta, device=rays.device)

    def apply_transform_(self, points):
        """
        Transform the coordinates of ``points`` in world-space to coordinates in object-space

        :param points: Vectors in world-space (:obj:`torch.tensor`)

        :return: New vectors in object-space (:obj:`torch.tensor`)
        """
        return torch.matmul(self.transform.to(points.device),
                            torch.cat((points.type(torch.double), torch.ones((points.shape[0], 1), dtype=torch.double,
                                                                             device=points.device)), dim=1).unsqueeze(
                                -1))[:, :3, 0].type(points.dtype)

    def apply_inverse_transform_(self, points):
        """
        Transform the coordinates of ``points`` in object-space to coordinates in world-space

        :param points: Vectors in object-space (:obj:`torch.tensor`)

        :return: New vectors in world-space (:obj:`torch.tensor`)
        """
        return torch.matmul(self.inverse_transform.to(points.device),
                            torch.cat((points.type(torch.double), torch.ones((points.shape[0], 1), dtype=torch.double,
                                                                             device=points.device)), dim=1).unsqueeze(
                                -1))[:, :3, 0].type(points.dtype)

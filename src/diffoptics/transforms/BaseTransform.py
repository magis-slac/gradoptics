import abc  # Abstract Base Classes
import torch
from diffoptics.optics.Ray import Rays


class BaseTransform(abc.ABC):

    def apply_transform(self, rays: Rays):
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

    def apply_inverse_transform(self, rays: Rays):
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

    def apply_transform_(self, points: torch.tensor):
        return torch.matmul(self.transform.to(points.device),
                            torch.cat((points.type(torch.double), torch.ones((points.shape[0], 1), dtype=torch.double,
                                                                             device=points.device)), dim=1).unsqueeze(
                                -1))[:, :3, 0].type(points.dtype)

    def apply_inverse_transform_(self, points: torch.tensor):
        return torch.matmul(self.inverse_transform.to(points.device),
                            torch.cat((points.type(torch.double), torch.ones((points.shape[0], 1), dtype=torch.double,
                                                                             device=points.device)), dim=1).unsqueeze(
                                -1))[:, :3, 0].type(points.dtype)

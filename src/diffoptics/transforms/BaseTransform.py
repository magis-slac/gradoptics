import abc  # Abstract Base Classes
import torch
from diffoptics.optics.Ray import Rays


class BaseTransform(abc.ABC):

    def apply_transform(self, rays: Rays):
        new_o = torch.matmul(self.transform.type(rays.origins.dtype).to(rays.device),
                             torch.cat((rays.origins, torch.ones((rays.origins.shape[0], 1),
                                        device=rays.device)), dim=1).unsqueeze(-1))[:, :3, 0]
        new_d = torch.matmul(self.transform.type(rays.directions.dtype).to(rays.device),
                             torch.cat((rays.directions, torch.zeros((rays.directions.shape[0], 1),
                                        device=rays.device)), dim=1).unsqueeze(-1))[:, :3, 0]
        return Rays(new_o, new_d, luminosities=rays.luminosities, meta=rays.meta, device=rays.device)

    def inverse_transform(self, rays: Rays):
        new_o = torch.matmul(self.inverse_transform.type(rays.origins.dtype).to(rays.device),
                             torch.cat((rays.origins, torch.ones((rays.origins.shape[0], 1),
                                        device=rays.device)), dim=1).unsqueeze(-1))[:, :3, 0]
        new_d = torch.matmul(self.inverse_transform.type(rays.directions.dtype).to(rays.device),
                             torch.cat((rays.directions, torch.zeros((rays.directions.shape[0], 1),
                                        device=rays.device)), dim=1).unsqueeze(-1))[:, :3, 0]
        return Rays(new_o, new_d, luminosities=rays.luminosities, meta=rays.meta, device=rays.device)

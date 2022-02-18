import torch

from diffoptics.optics.Vector import normalize_vector
from diffoptics.optics.Vector import cross_product
from diffoptics.transforms.BaseTransform import BaseTransform


class SimpleTransform(BaseTransform):

    def __init__(self, theta_x: float, theta_y: float, theta_z: float, t: torch.tensor):

        M_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(torch.tensor([theta_x])), -torch.sin(torch.tensor([theta_x]))],
                            [0, torch.sin(torch.tensor([theta_x])), torch.cos(torch.tensor([theta_x]))]],
                           dtype=torch.double)

        M_y = torch.tensor([[torch.cos(torch.tensor([theta_y])), 0, -torch.sin(torch.tensor([theta_y]))],
                            [0, 1, 0],
                            [torch.sin(torch.tensor([theta_y])), 0, torch.cos(torch.tensor([theta_y]))]],
                           dtype=torch.double)

        M_z = torch.tensor([[torch.cos(torch.tensor([theta_z])), -torch.sin(torch.tensor([theta_z])), 0],
                            [torch.sin(torch.tensor([theta_z])), torch.cos(torch.tensor([theta_z])), 0],
                            [0, 0, 1]],
                           dtype=torch.double)
        M = M_z @ M_y @ M_x

        self.transform = torch.tensor([[M[0, 0], M[0, 1], M[0, 2], t[0]],
                                       [M[1, 0], M[1, 1], M[1, 2], t[1]],
                                       [M[2, 0], M[2, 1], M[2, 2], t[2]],
                                       [0, 0, 0, 1]], dtype=torch.double)
        self.inverse_transform = torch.inverse(self.transform)

        # Sanity check (making sure that the inversion went well)
        assert torch.allclose(torch.matmul(self.transform, self.inverse_transform), torch.eye(4, dtype=torch.double),
                              rtol=1e-04, atol=1e-06)
        x = torch.randn((4, 4), dtype=torch.double)
        assert torch.allclose(torch.matmul(self.transform, torch.matmul(self.inverse_transform, x)), x, rtol=1e-04,
                              atol=1e-06)
        assert torch.allclose(torch.matmul(self.transform, torch.matmul(self.inverse_transform, x)), x, rtol=1e-04,
                              atol=1e-06)

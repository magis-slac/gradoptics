import torch

from diffoptics.optics.Vector import normalize_vector
from diffoptics.optics.Vector import cross_product
from diffoptics.transforms.BaseTransform import BaseTransform


class SimpleTransform(BaseTransform):

    def __init__(self, theta_x: float, theta_y: float, theta_z: float, t: torch.tensor):
        t = torch.tensor([[1, 0, 0, t[0]],
                          [0, 1, 0, t[1]],
                          [0, 0, 1, t[2]],
                          [0, 0, 0, 1]])

        M_x = torch.tensor([[1, 0, 0, 0],
                            [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
                            [0, torch.sin(theta_x), torch.cos(theta_x), 0],
                            [0, 0, 0, 1]])

        M_y = torch.tensor([[torch.cos(theta_y), 0, -torch.sin(theta_y), 0],
                            [0, 1, 0, 0],
                            [torch.sin(theta_y), 0, torch.cos(theta_y), 0],
                            [0, 0, 0, 1]])

        M_z = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
                            [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        self.transform = M_z @ M_y @ M_x @ t
        self.inverse_transform = torch.inverse(self.transform)

        # Sanity check (making sure that the inversion went well)
        assert torch.allclose(torch.matmul(self.transform, self.inverse_transform), torch.eye(4), rtol=1e-04, atol=1e-06
                              )
        x = torch.randn(4, 4)
        assert torch.allclose(torch.matmul(self.transform, torch.matmul(self.inverse_transform, x)), x, rtol=1e-04,
                              atol=1e-06)
        assert torch.allclose(torch.matmul(self.transform, torch.matmul(self.inverse_transform, x)), x, rtol=1e-04,
                              atol=1e-06)

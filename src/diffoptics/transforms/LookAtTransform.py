import torch

from diffoptics.optics.Vector import normalize_vector
from diffoptics.optics.Vector import cross_product
from diffoptics.transforms.BaseTransform import BaseTransform


class LookAtTransform(BaseTransform):

    def __init__(self, viewing_direction: torch.tensor, pos: torch.tensor, up=torch.tensor([0., 0., 1.])):
        """
        Given a viewing direction in a world-space, returns the
        4x4 transform matrices to move a point from camera-space to world-space and
        from world-space to camera-space
        References: Physically based rendering, section 2.7.7 The look-at transform
        :param viewing_direction: viewing direction of the camera
        :param pos: position of the camera
        :param up: a vector that orients the camera with respect to the viewing direction.
                   For example, if up=torch.tensor([0, 0, 1]) the top of the camera will point upwards.
                   If up=torch.tensor([0, 0, -1]), the top of the camera will point downwards.
        :return:
        """
        dir_ = normalize_vector(viewing_direction)
        left = normalize_vector(cross_product(normalize_vector(up), dir_))
        new_up = cross_product(dir_, left)

        self.transform = torch.tensor([[left[0], new_up[0], dir_[0], pos[0]],
                                       [left[1], new_up[1], dir_[1], pos[1]],
                                       [left[2], new_up[2], dir_[2], pos[2]],
                                       [0., 0., 0., 1.]], dtype=torch.double)
        self.inverse_transform = torch.inverse(self.transform)

        # Sanity check (making sure that the inversion went well)
        assert torch.allclose(torch.matmul(self.transform, self.inverse_transform), torch.eye(4, dtype=torch.double),
                              rtol=1e-04, atol=1e-06)
        x = torch.randn((4, 4), dtype=torch.double)
        assert torch.allclose(torch.matmul(self.transform, torch.matmul(self.inverse_transform, x)), x, rtol=1e-04,
                              atol=1e-06)
        assert torch.allclose(torch.matmul(self.transform, torch.matmul(self.inverse_transform, x)), x, rtol=1e-04,
                              atol=1e-06)

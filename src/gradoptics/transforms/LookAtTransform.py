import torch

from gradoptics.optics.Vector import normalize_vector
from gradoptics.optics.Vector import cross_product
from gradoptics.transforms.BaseTransform import BaseTransform


class LookAtTransform(BaseTransform):
    """
    Transform that orients an object so that it is oriented towards a given direction.

    References: Physically based rendering, section 2.7.7 The look-at transform.
    """

    def __init__(self, viewing_direction, pos, up=torch.tensor([0., 0., 1.])):
        """
        Given a viewing direction in world-space, computes the 4x4 transform matrix to move a point from object-space
        to world-space and from world-space to object-space

        :param viewing_direction: viewing direction of the object (:obj:`torch.tensor`)
        :param pos: position of the object (:obj:`torch.tensor`)
        :param up: a vector that orients the object with respect to the viewing direction (:obj:`torch.tensor`).
                   For example, if up=torch.tensor([0, 0, 1]) the top of the object will point upwards.
                   If up=torch.tensor([0, 0, -1]), the top of the object will point downwards.
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

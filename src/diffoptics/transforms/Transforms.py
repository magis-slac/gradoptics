import torch

from diffoptics.optics.Vector import normalize_vector
from diffoptics.optics.Vector import cross_product


def get_look_at_transform(viewing_direction: torch.tensor, pos: torch.tensor, up=torch.tensor([0, 0, 1])):
    """
    Given a viewing direction in a left-handed coordinate system, returns the
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

    camera_to_world = torch.tensor([[left[0], new_up[0], dir_[0], pos[0]],
                                    [left[1], new_up[1], dir_[1], pos[1]],
                                    [left[2], new_up[2], dir_[2], pos[2]],
                                    [0, 0, 0, 1]])
    world_to_camera = torch.inverse(camera_to_world)

    # Sanity check (making sure that the inversion went well)
    assert (torch.matmul(camera_to_world, world_to_camera) == torch.eye(4)).all()
    x = torch.randn(4, 4)
    assert torch.allclose(torch.matmul(camera_to_world, torch.matmul(world_to_camera, x)), x)
    assert torch.allclose(torch.matmul(world_to_camera, torch.matmul(camera_to_world, x)), x)

    return camera_to_world, world_to_camera

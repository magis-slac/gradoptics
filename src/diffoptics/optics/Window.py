import torch

from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.optics.Ray import Rays
from diffoptics.optics.Vector import dot_product


class Window(BaseOptics):

    # Assumes that the window is on the plane x=x'. Generalization may be required for other usages.

    def __init__(self, left_interface_x_position, right_interface_x_position, n_ext=1.000293, n_glass=1.494,
                 diameter=0.137, eps=1e-15):

        super(Window, self).__init__()
        assert right_interface_x_position > left_interface_x_position

        self.diameter = diameter
        self.right_interface_x_position = right_interface_x_position
        self.left_interface_x_position = left_interface_x_position
        self.n_ext = n_ext
        self.n_glass = n_glass
        self.eps = eps

    def _get_ray_intersection_left_interface(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the left interface of the window
        :param incident_rays:
        :return:
        """
        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (self.left_interface_x_position - origins[:, 0]) / (directions[:, 0] + self.eps)
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]
        # Check that the intersection is real (t>0) and within the windows' aperture
        condition = (t > 0) & ((y ** 2 + z ** 2) < (self.diameter / 2) ** 2)
        t[~condition] = float('inf')
        return t

    @torch.no_grad()
    def _get_ray_intersection_right_interface(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the right interface of the window
        :param incident_rays:
        :return:
        """
        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (self.right_interface_x_position - origins[:, 0]) / (directions[:, 0] + self.eps)
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]
        # Check that the intersection is real (t>0) and within the windows' aperture
        condition = (t > 0) & ((y ** 2 + z ** 2) < (self.diameter / 2) ** 2)
        t[~condition] = float('inf')
        return t

    @torch.no_grad()
    def get_ray_intersection(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the window
        :param incident_rays:
        :return:
        """
        t_left = self._get_ray_intersection_left_interface(incident_rays)
        t_right = self._get_ray_intersection_right_interface(incident_rays)
        # Keep the smallest t
        t = torch.empty(t_left.shape, device=t_left.device)
        condition = t_right < t_left
        t[condition] = t_right[condition]
        t[~condition] = t_left[~condition]
        t[torch.isinf(t)] = float('nan')
        return t

    def intersect(self, incident_rays, t):
        """
        Returns the ray refracted by the window
        @Todo
        Note: modifies incident_rays
        :param incident_rays:
        :param t:
        :return:
        """

        origins = incident_rays.origins
        directions = incident_rays.directions
        origin_refracted_rays = origins + t.unsqueeze(1) * directions

        # Interaction with the first interface
        window_normal = torch.zeros(directions.shape, device=origins.device)
        window_normal[:, 0] = 1
        # Check for each ray if it is coming from the left
        condition = directions[:, 0] > 0
        # Flip the normal for the rays coming from the right
        window_normal[~condition] *= -1
        mu = self.n_ext / self.n_glass
        # See https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
        tmp = 1 - mu ** 2 * (1 - (dot_product(window_normal, directions)) ** 2)
        c = dot_product(window_normal, directions)
        direction_refracted_rays = torch.empty((tmp.shape[0], 3), device=origins.device)
        direction_refracted_rays[:, 0] = torch.sqrt(tmp) * window_normal[:, 0] + mu * (
                directions[:, 0] - c * window_normal[:, 0])
        direction_refracted_rays[:, 1] = torch.sqrt(tmp) * window_normal[:, 1] + mu * (
                directions[:, 1] - c * window_normal[:, 1])
        direction_refracted_rays[:, 2] = torch.sqrt(tmp) * window_normal[:, 2] + mu * (
                directions[:, 2] - c * window_normal[:, 2])
        # Checks
        theta_1 = torch.arccos(directions[:, 0] * window_normal[:, 0] +
                               directions[:, 1] * window_normal[:, 1] +
                               directions[:, 2] * window_normal[:, 2])
        theta_2 = torch.arccos(direction_refracted_rays[:, 0] * window_normal[:, 0] +
                               direction_refracted_rays[:, 1] * window_normal[:, 1] +
                               direction_refracted_rays[:, 2] * window_normal[:, 2])
        assert ((torch.sin(theta_1) * self.n_ext - torch.sin(
            theta_2) * self.n_glass).abs().sum() / theta_1.shape[0]) < 1e-5
        ray_in_glass = Rays(origin_refracted_rays,
                            direction_refracted_rays,
                            luminosities=incident_rays.luminosities,
                            device=incident_rays.device)

        # Interaction with the second interface
        t = self.get_ray_intersection(ray_in_glass)
        ray_in_glass = ray_in_glass[~torch.isnan(t)]
        incident_rays.origins = incident_rays.origins[~torch.isnan(t)]
        incident_rays.directions = incident_rays.directions[~torch.isnan(t)]
        window_normal = window_normal[~torch.isnan(t)]
        t = t[~torch.isnan(t)]
        origins = ray_in_glass.origins
        directions = ray_in_glass.directions
        origin_refracted_rays = origins + t.unsqueeze(1) * directions
        mu = self.n_glass / self.n_ext
        tmp = 1 - mu ** 2 * (1 - (dot_product(window_normal, directions)) ** 2)
        c = dot_product(window_normal, directions)
        direction_refracted_rays = torch.empty((tmp.shape[0], 3), device=origins.device)
        direction_refracted_rays[:, 0] = torch.sqrt(tmp) * window_normal[:, 0] + mu * (
                    directions[:, 0] - c * window_normal[:, 0])
        direction_refracted_rays[:, 1] = torch.sqrt(tmp) * window_normal[:, 1] + mu * (
                    directions[:, 1] - c * window_normal[:, 1])
        direction_refracted_rays[:, 2] = torch.sqrt(tmp) * window_normal[:, 2] + mu * (
                    directions[:, 2] - c * window_normal[:, 2])
        # Checks
        theta_1 = torch.arccos(directions[:, 0] * window_normal[:, 0] +
                               directions[:, 1] * window_normal[:, 1] +
                               directions[:, 2] * window_normal[:, 2])
        theta_2 = torch.arccos(direction_refracted_rays[:, 0] * window_normal[:, 0] +
                               direction_refracted_rays[:, 1] * window_normal[:, 1] +
                               direction_refracted_rays[:, 2] * window_normal[:, 2])
        assert ((torch.sin(theta_1) * self.n_glass - torch.sin(theta_2) * self.n_ext).abs().sum() / theta_1.shape[
            0]) < 1e-5
        return Rays(origin_refracted_rays, direction_refracted_rays, luminosities=incident_rays.luminosities,
                    meta=incident_rays.meta, device=incident_rays.device)

    def plot(self, ax, s=0.1, color='lightblue', resolution=100):
        # @Todo, change this to plot_surface
        y = torch.arange(-self.diameter / 2, self.diameter / 2, self.diameter / resolution)
        z = torch.arange(-self.diameter / 2, self.diameter / 2, self.diameter / resolution)
        y, z = torch.meshgrid(y, z)

        y = y.reshape(-1)
        z = z.reshape(-1)
        x = torch.zeros(resolution * resolution)

        indices = (y ** 2 + z ** 2) < (self.diameter / 2) ** 2
        x_left = x[indices] + self.left_interface_x_position
        x_right = x[indices] + self.right_interface_x_position
        y = y[indices]
        z = z[indices]

        ax.scatter(x_left, y, z, s=s, c=color)
        ax.scatter(x_right, y, z, s=s, c=color)

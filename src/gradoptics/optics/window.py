import torch

from gradoptics.optics.base_optics import BaseOptics
from gradoptics.optics.ray import Rays
from gradoptics.optics.vector import dot_product
from gradoptics.transforms.simple_transform import SimpleTransform


class Window(BaseOptics):
    """
    Models a medium with a refractive index between two parallel surfaces. Refraction is modelled using Snell's law.
    """

    def __init__(self, left_interface_x_position, right_interface_x_position, n_ext=1.000293, n_glass=1.494,
                 diameter=0.137, eps=1e-15, orientation=(0., 0., 0.)):
        """
        :param left_interface_x_position: Position of the left interface along the x axis (:obj:`float`)
        :param right_interface_x_position: Position of the right interface along the x axis (:obj:`float`)
        :param n_ext: Refractive index of the external medium (:obj:`float`)
        :param n_glass: Refractive index of the medium (:obj:`float`)
        :param diameter: Diameter of the interfaces (:obj:`float`)
        :param eps: Parameter used for numerical stability in the different class methods (:obj:`float`). Default
                    is ``'1e-15'``
        """

        super(Window, self).__init__()
        assert right_interface_x_position > left_interface_x_position

        self.diameter = diameter
        self.right_interface_x_position = right_interface_x_position
        self.left_interface_x_position = left_interface_x_position
        self.n_ext = n_ext
        self.n_glass = n_glass
        self.eps = eps
        self.w2obj = SimpleTransform(*orientation, torch.zeros(3))
        
    def fresnel_coeff(self, theta, n1, n2):
        """
        Fresnel transmission coefficient
        :param cos_theta: cos(theta) between window normal and ray direction
        """
        num_s = n1*torch.cos(theta)-n2*torch.sqrt(1-(n1/n2*torch.sin(theta))**2)
        den_s = n1*torch.cos(theta)+n2*torch.sqrt(1-(n1/n2*torch.sin(theta))**2)
        r_s = abs(num_s/den_s)**2
        
        num_p = n1*torch.sqrt(1-(n1/n2*torch.sin(theta))**2)-n2*torch.cos(theta)
        den_p = n1*torch.sqrt(1-(n1/n2*torch.sin(theta))**2)+n2*torch.cos(theta)
        
        r_p = abs(num_p/den_p)**2

        return 1-0.5*(r_s+r_p) 

    def _get_ray_intersection_left_interface(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the left interface of the window
        :param incident_rays:
        :return:
        """
        # Rotate to coords with window along positive x
        incident_rays = self.w2obj.apply_inverse_transform(incident_rays) 

        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (self.left_interface_x_position - origins[:, 0]) / (directions[:, 0] + self.eps)
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]
        # Check that the intersection is real (t>0) and within the windows' aperture
        condition = (t > 1e-3) & ((y ** 2 + z ** 2) < (self.diameter / 2) ** 2)
        t[~condition] = float('inf')
        return t

    @torch.no_grad()
    def _get_ray_intersection_right_interface(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the right interface of the window
        :param incident_rays:
        :return:
        """
        # Rotate to coords with window along positive x
        incident_rays = self.w2obj.apply_inverse_transform(incident_rays) 

        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (self.right_interface_x_position - origins[:, 0]) / (directions[:, 0] + self.eps)
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]
        # Check that the intersection is real (t>0) and within the windows' aperture
        condition = (t > 1e-3) & ((y ** 2 + z ** 2) < (self.diameter / 2) ** 2)
        t[~condition] = float('inf')
        return t

    def _get_rays_inside_window(self, incident_rays, t, real_intersect=False):
        
        # Rotate to coords with window along positive x
        incident_rays = self.w2obj.apply_inverse_transform(incident_rays)

        origins = incident_rays.origins
        directions = incident_rays.directions
        origin_refracted_rays = origins + t.unsqueeze(1) * directions

        # Interaction with the first interface
        window_normal = torch.zeros(directions.shape, device=directions.device, dtype=directions.dtype)
        window_normal[:, 0] = 1
        # Check for each ray if it is coming from the left
        condition = directions[:, 0] > 0
        # Flip the normal for the rays coming from the right
        window_normal[~condition] *= -1
        mu = self.n_ext / self.n_glass
        # See https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
        tmp = 1 - mu ** 2 * (1 - (dot_product(window_normal, directions)) ** 2)
        c = dot_product(window_normal, directions)
        direction_refracted_rays = torch.sqrt(tmp).unsqueeze(1) * window_normal + mu * (
                directions - c.unsqueeze(1) * window_normal)

        trans_coeff = self.fresnel_coeff(torch.acos(c), self.n_ext, self.n_glass)
        output_rays = Rays(origin_refracted_rays, direction_refracted_rays, luminosities=incident_rays.luminosities*trans_coeff,
                    device=incident_rays.device)

        # Rotate to back to world coords
        output_rays = self.w2obj.apply_transform(output_rays) 

        window_normal = torch.matmul(self.w2obj.transform.to(window_normal.device),
                             torch.cat((window_normal.type(torch.double),
                                        torch.zeros((window_normal.shape[0], 1), dtype=torch.double,
                                                    device=window_normal.device)), dim=1).unsqueeze(-1))[:, :3, 0].type(
                                    window_normal.dtype)
                    
        return output_rays, window_normal

    def _get_t_min(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the window
        :param incident_rays:
        :return:
        """
        t_left = self._get_ray_intersection_left_interface(incident_rays)
        t_right = self._get_ray_intersection_right_interface(incident_rays)
        # Keep the smallest t
        t = torch.empty(t_left.shape, device=t_left.device, dtype=t_left.dtype)
        condition = t_right < t_left
        t[condition] = t_right[condition]
        t[~condition] = t_left[~condition]
        t[torch.isinf(t)] = float('nan')
        return t

    def get_ray_intersection(self, incident_rays):

        # Time t at which the rays will intersect the first interface
        t1 = self._get_t_min(incident_rays)

        # Only keep the rays that have intersected the first interface
        mask = ~torch.isnan(t1)

        # Return if no rays have intersected the first interface
        if mask.sum() == 0:
            return t1

        ray_in_glass, window_normal = self._get_rays_inside_window(incident_rays[mask], t1[mask])
        # Interaction with the second interface
        t2 = self._get_t_min(ray_in_glass)
        nan_mask = torch.isnan(t2)
        t2[~nan_mask] = t1[mask][~nan_mask]  # t2 should only overwrite t1 where it is nan (rays that are lost)
        t1[mask] = t2
        return t1

    def intersect(self, incident_rays, t):

        ray_in_glass, window_normal = self._get_rays_inside_window(incident_rays, t, real_intersect=True)

        # Interaction with the second interface
        t = self._get_t_min(ray_in_glass)
        ray_in_glass = ray_in_glass[~torch.isnan(t)]
        incident_rays.origins = incident_rays.origins[~torch.isnan(t)]
        incident_rays.directions = incident_rays.directions[~torch.isnan(t)]
        incident_rays.luminosities = incident_rays.luminosities[~torch.isnan(t)]
        for key in incident_rays.meta.keys():
            incident_rays.meta[key] = incident_rays.meta[key][~torch.isnan(t)]
        window_normal = window_normal[~torch.isnan(t)]
        t = t[~torch.isnan(t)]
        origins = ray_in_glass.origins
        directions = ray_in_glass.directions
        origin_refracted_rays = origins + t.unsqueeze(1) * directions
        mu = self.n_glass / self.n_ext
        tmp = 1 - mu ** 2 * (1 - (dot_product(window_normal, directions)) ** 2)
        c = dot_product(window_normal, directions)
        direction_refracted_rays = torch.sqrt(tmp).unsqueeze(1) * window_normal + mu * (
                directions - c.unsqueeze(1) * window_normal)
        trans_coeff = self.fresnel_coeff(torch.acos(c), self.n_glass, self.n_ext)
        return (Rays(origin_refracted_rays, direction_refracted_rays, luminosities=incident_rays.luminosities*trans_coeff,
                    meta=incident_rays.meta, device=incident_rays.device),
                torch.ones(origin_refracted_rays.shape[0], dtype=torch.bool, device=origin_refracted_rays.device))

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

        x_left, y_left, z_left = self.w2obj.apply_transform_(torch.cat((x_left[:, None], y[:, None], z[:, None]), dim=1)).T
        x_right, y_right, z_right = self.w2obj.apply_transform_(torch.cat((x_right[:, None], y[:, None], z[:, None]), dim=1)).T
        ax.scatter(x_left, y_left, z_left, s=s, c=color)
        ax.scatter(x_right, y_right, z_right, s=s, c=color)

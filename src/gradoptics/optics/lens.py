import abc
import torch
import numpy as np
import gradoptics as optics

from gradoptics.optics.base_optics import BaseOptics
from gradoptics.optics.ray import Rays
from gradoptics.optics.vector import batch_vector
from gradoptics.transforms.simple_transform import SimpleTransform


class Lens(BaseOptics, abc.ABC):
    """
    Base class for lenses.
    """

    @abc.abstractmethod
    def sample_points_on_lens(self, nb_points, device='cpu'):
        """
        Sample points uniformly on the lens

        :param nb_points: Number of points to sample (:obj:`int`)
        :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

        :return: Sampled points (:obj:`torch.tensor`)
        """
        raise NotImplemented


class PerfectLens(Lens):
    """
    Models a thin lens.
    """

    def __init__(self, f=0.05, na=0.714, position=None, m=0.15, transform=None, eps=1e-15):
        """
        :param f: Focal length (:obj:`float`)
        :param na: Inverse of the f-number (:obj:`float`)
        :param position: Position of the lens (:obj:`list`)
        :param m: Lens magnification  (:obj:`float`)
        :param transform: Transform to orient the lens (:py:class:`~gradoptics.transforms.BaseTransform.BaseTransform`)
        :param eps: Parameter used for numerical stability in the different class methods (:obj:`float`). Default
                    is ``'1e-15'``
        """
        super(PerfectLens, self).__init__()

        if position is None:
            position = [0., 0., 0.]

        self.f = f
        self.na = na
        self.m = m
        self.pof = f * (m + 1) / m
        self.eps = eps
        if transform is None:
            self.transform = SimpleTransform(0., 0., 0., torch.tensor(position))
        else:
            self.transform = transform

    def get_ray_intersection(self, incident_rays):
        incident_rays = self.transform.apply_inverse_transform(incident_rays)  # World space to lens space
        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (- origins[:, 0]) / (directions[:, 0] + self.eps)
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]

        # Check that the intersection is real (t>0) and within the lens' bounds
        condition = (t > 0) & ((y ** 2 + z ** 2) < (self.f * self.na / 2) ** 2)
        t[~condition] = float('nan')
        return t

    def intersect(self, incident_rays, t):
        incident_rays = self.transform.apply_inverse_transform(incident_rays)  # World space to lens space

        x_objects = incident_rays.origins
        d_0 = incident_rays.directions

        # If x_objects is on the left of the lens, put the plane of focus on the left as well
        pof = self.pof * torch.sign(x_objects[:, 0])
        # Put the camera on the opposite side of the lens with respect to x_objects
        camera_pos = - self.f * (1 + self.m) * torch.sign(x_objects[:, 0])

        # Intersections with the lens
        x_0 = x_objects + t.unsqueeze(1) * d_0

        # computes x_star, the intersections of the rays with the plane of focus
        t = (x_0[:, 0] - pof) / (d_0[:, 0] + self.eps)
        t = -t
        x_star = x_0 + t.unsqueeze(1) * d_0

        # Computes x_v, the intersections of the rays coming from x_star and passing trough the optical center with the
        # camera
        d = - x_star
        t = (camera_pos - x_star[:, 0]) / (d[:, 0] + self.eps)
        x_v = x_star + t.unsqueeze(1) * d
        _d_out = batch_vector(x_v[:, 0] - x_0[:, 0], x_v[:, 1] - x_0[:, 1], x_v[:, 2] - x_0[:, 2])

        return (self.transform.apply_transform(Rays(x_0, _d_out, luminosities=incident_rays.luminosities,
                                                    meta=incident_rays.meta, device=incident_rays.device)),
                torch.ones(x_0.shape[0], dtype=torch.bool, device=x_0.device))

    def sample_points_on_lens(self, nb_points, device='cpu'):

        points = torch.zeros((nb_points, 3), device=device)
        lens_radius = self.f * self.na / 2
        # Sampling uniformly on a disk.
        # See https://stats.stackexchange.com/questions/481543/generating-random-points-uniformly-on-a-disk
        r_squared = torch.rand(nb_points, device=device) * lens_radius ** 2
        theta = torch.rand(nb_points, device=device) * 2 * np.pi
        points[:, 1] = torch.sqrt(r_squared) * torch.cos(theta)
        points[:, 2] = torch.sqrt(r_squared) * torch.sin(theta)
        return self.transform.apply_transform_(points)

    def plot(self, ax, s=0.1, color='lightblue', resolution=100):
        # @Todo, change this to plot_surface
        y = torch.arange(-self.f * self.na / 2, self.f * self.na / 2, (self.f * self.na) / resolution)
        z = torch.arange(-self.f * self.na / 2, self.f * self.na / 2, (self.f * self.na) / resolution)
        y, z = torch.meshgrid(y, z)

        y = y.reshape(-1)
        z = z.reshape(-1)
        x = torch.zeros(resolution * resolution)

        indices = (y ** 2 + z ** 2) < (self.f * self.na / 2) ** 2
        x = x[indices]
        y = y[indices]
        z = z[indices]

        # Lens space to world space
        xyz = self.transform.apply_transform_(torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1))

        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=s, c=color)


class ThickLens(Lens):
    """
    Models a thick lens using Snell's law.
    """

    def __init__(self, n_lens, n_ext, R, d, transform):
        """
        :param n_lens: Index of refraction of the lens (:obj:`float`)
        :param n_ext: Index of refraction of the external medium (:obj:`float`)
        :param R: Radius of curvature of the lens (:obj:`float`)
        :param d: Thickness of the lens (:obj:`float`)
        :param transform: Transform to orient the lens (:py:class:`~gradoptics.transforms.BaseTransform.BaseTransform`)
        """
        super(ThickLens, self).__init__()
        self.n_lens = n_lens
        self.n_ext = n_ext
        self.R = R
        self.d = d
        self.transform = transform

    def get_focal_length(self):
        """
        Approximates the focal length of the lens based on its properties

        :return: Approximation of the focal length (:obj:`float`)
        """
        R1 = self.R
        R2 = - self.R
        return 1 / ((self.n_ext / self.n_lens - 1) * (1 / R1 - 1 / R2))

    def get_normal(self, points):
        """
        Return the normal of the lens at the given points. The normals are oriented towards the center of the lens

        :param points: 3d positions (:obj:`torch.tensor`)

        :return: Normals at the given points (:obj:`torch.tensor`)
        """

        # Computes the center of the spheres on which the points are
        center = torch.zeros_like(points)
        # -R or R as a function of the surface on which the points are
        center[:, 0] = - torch.sign(points[:, 0]) * self.R

        return optics.normalize_batch_vector(center - points)

    @staticmethod
    def _compute_intersection_with_sphere(incident_rays, R, sphere_center):
        """
        Computes the intersection of the rays ``incident_rays`` with a sphere of radius ``R``, and center
        ``sphere_center``

        :param incident_rays: The incident rays (:obj:`torch.tensor`)
        :param R: Radius of the sphere (:obj:`float`)
        :param sphere_center: Center of the sphere (:obj:`list`)

        :return:
        """
        origins, directions = incident_rays.get_origin_and_direction()

        # Solve quadratic equation in t (intersection between the incident_rays and the sphere)
        a = directions[:, 0] ** 2 + directions[:, 1] ** 2 + directions[:, 2] ** 2
        b = 2 * (((origins[:, 0] - sphere_center[0]) * directions[:, 0]) + (
                (origins[:, 1] - sphere_center[1]) * directions[:, 1]) + (
                         (origins[:, 2] - sphere_center[2]) * directions[:, 2]))
        c = (origins[:, 0] - sphere_center[0]) ** 2 + (origins[:, 1] - sphere_center[1]) ** 2 + (
                origins[:, 2] - sphere_center[2]) ** 2 - R ** 2

        pho = b ** 2 - 4 * a * c

        t1 = torch.zeros(origins.shape[0])
        t2 = torch.zeros(origins.shape[0])
        mask = pho > 0
        t1[~mask] = float('Inf')
        t2[~mask] = float('Inf')

        sqrt_pho = np.sqrt(pho[mask])
        t1[mask] = (-b[mask] + sqrt_pho) / (2 * a[mask])
        t2[mask] = (-b[mask] - sqrt_pho) / (2 * a[mask])

        # Remove virtual intersections
        t1[t1 < 0] = float('Inf')
        t2[t2 < 0] = float('Inf')
        return t1, t2

    def _get_ray_intersection_left_surface(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the left surface of the lens

        :param incident_rays: The incident rays (:py:class:`~gradoptics.optics.Ray.Rays`)

        :return: Times t (:obj:`torch.tensor`)
        """

        origin_sphere1 = [-self.R + self.d / 2, 0., 0.]
        t1, t2 = self._compute_intersection_with_sphere(incident_rays, self.R, origin_sphere1)
        # Only a portion of the surface makes up the lens. Remove intersection with other sections of the sphere
        t1[incident_rays(t1)[:, 0] < 0] = float('Inf')
        t2[incident_rays(t2)[:, 0] < 0] = float('Inf')
        return torch.min(t1, t2)

    def _get_ray_intersection_right_surface(self, incident_rays):
        """
        Computes the times t at which the incident rays will intersect the right surface of the lens

        :param incident_rays: The incident rays (:py:class:`~gradoptics.optics.Ray.Rays`)

        :return: Times t (:obj:`torch.tensor`)
        """

        # Intersection with second surface
        origin_sphere2 = [self.R - self.d / 2, 0., 0.]
        t1, t2 = self._compute_intersection_with_sphere(incident_rays, self.R, origin_sphere2)
        # Only a portion of the surface makes up the lens. Remove intersection with other sections of the sphere
        t1[incident_rays(t1)[:, 0] > 0] = float('Inf')
        t2[incident_rays(t2)[:, 0] > 0] = float('Inf')
        return torch.min(t1, t2)

    def _get_rays_inside_lens(self, incident_rays, t):
        """
        Computes the rays refracted by the first surface

        :param incident_rays: The incident rays (:py:class:`~gradoptics.optics.Ray.Rays`)
        :param t: The times at which the incident rays will intersect the optical element (:obj:`torch.tensor`)

        :return: (:obj:`tuple`)
                 - Reflected or refracted rays (:py:class:`~gradoptics.optics.Ray.Rays`)
                 - A boolean mask that indicates which incident rays were refracted ― some rays may not be
                   refracted (:obj:`torch.tensor`)
        """

        directions = incident_rays.directions
        origin_refracted_rays = incident_rays(t)

        normals = self.get_normal(incident_rays(t))
        mu = self.n_ext / self.n_lens
        # See https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
        tmp = 1 - mu ** 2 * (1 - (optics.dot_product(normals, directions)) ** 2)
        c = optics.dot_product(normals, directions)

        total_internal_refraction_mask = tmp < 0
        mask = ~total_internal_refraction_mask

        direction_refracted_rays = torch.sqrt(tmp[mask]).unsqueeze(1) * normals[mask] + mu * (
                directions[mask] - c[mask].unsqueeze(1) * normals[mask])

        # Sanity check (should be removed)
        theta_1 = torch.arccos(
            directions[~total_internal_refraction_mask, 0] * normals[~total_internal_refraction_mask, 0] +
            directions[~total_internal_refraction_mask, 1] * normals[~total_internal_refraction_mask, 1] +
            directions[~total_internal_refraction_mask, 2] * normals[~total_internal_refraction_mask, 2])
        theta_2 = torch.arccos(direction_refracted_rays[:, 0] * normals[~total_internal_refraction_mask, 0] +
                               direction_refracted_rays[:, 1] * normals[~total_internal_refraction_mask, 1] +
                               direction_refracted_rays[:, 2] * normals[~total_internal_refraction_mask, 2])
        if theta_1.shape[0] > 0:
            assert ((torch.sin(theta_1) * self.n_ext - torch.sin(theta_2) * self.n_lens).abs().mean()) < 1e-5

        return (Rays(origin_refracted_rays[mask], direction_refracted_rays,
                     luminosities=incident_rays.luminosities[mask], device=incident_rays.device),
                mask)

    def get_ray_intersection(self, incident_rays):

        t_left = self._get_ray_intersection_left_surface(incident_rays)
        t_right = self._get_ray_intersection_right_surface(incident_rays)

        # Keep the smallest t
        t = torch.empty(t_left.shape, device=t_left.device, dtype=t_left.dtype)
        condition = t_right < t_left
        t[condition] = t_right[condition]
        t[~condition] = t_left[~condition]
        t[torch.isinf(t)] = float('nan')
        return t

    def intersect(self, incident_rays_, t_):

        incident_rays, mask = self._get_rays_inside_lens(incident_rays_, t_)

        interface_mask = incident_rays.origins[:, 0] < 0
        t = torch.empty(interface_mask.shape[0], device=mask.device)
        t[interface_mask] = self._get_ray_intersection_left_surface(incident_rays[interface_mask])
        t[~interface_mask] = self._get_ray_intersection_right_surface(incident_rays[~interface_mask])

        # A ray that was refracted by a surface may hit that surface again. Killing those rays
        mask_ = ~torch.isinf(t)

        incident_rays = incident_rays[mask_]
        t = t[mask_]

        # Interaction with the second surface
        directions = incident_rays.directions
        origin_refracted_rays = incident_rays(t)

        normal = -self.get_normal(incident_rays(t))
        mu = self.n_lens / self.n_ext
        # See https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
        tmp = 1 - mu ** 2 * (1 - (optics.dot_product(normal, directions)) ** 2)

        c = optics.dot_product(normal, directions)
        total_internal_refraction_mask = tmp < 0

        direction_refracted_rays = torch.sqrt(tmp).unsqueeze(1) * normal + mu * (directions - c.unsqueeze(1) * normal)

        # Sanity check (should be removed)
        theta_1 = torch.arccos(
            directions[~total_internal_refraction_mask, 0] * normal[~total_internal_refraction_mask, 0] +
            directions[~total_internal_refraction_mask, 1] * normal[~total_internal_refraction_mask, 1] +
            directions[~total_internal_refraction_mask, 2] * normal[~total_internal_refraction_mask, 2])
        theta_2 = torch.arccos(
            direction_refracted_rays[~total_internal_refraction_mask, 0] * normal[~total_internal_refraction_mask, 0] +
            direction_refracted_rays[~total_internal_refraction_mask, 1] * normal[~total_internal_refraction_mask, 1] +
            direction_refracted_rays[~total_internal_refraction_mask, 2] * normal[~total_internal_refraction_mask, 2])

        if theta_1.shape[0] > 0:
            assert ((torch.sin(theta_1) * self.n_lens - torch.sin(theta_2) * self.n_ext).abs().mean()) < 1e-5

        return Rays(origin_refracted_rays, direction_refracted_rays, luminosities=incident_rays.luminosities,
                    device=incident_rays.device), None

    def plot(self, ax, resolution=200, **kwargs):

        half_lens_height = np.sqrt(self.R ** 2 - (self.R - self.d / 2) ** 2)
        y = torch.arange(-half_lens_height, half_lens_height, 2 * half_lens_height / resolution)
        z = torch.arange(-half_lens_height, half_lens_height, 2 * half_lens_height / resolution)
        y, z = torch.meshgrid(y, z)

        y = y.reshape(-1)
        z = z.reshape(-1)
        x = np.sqrt(self.R ** 2 - y ** 2 - z ** 2)
        x = x - self.R + self.d / 2

        indices = x >= 0
        x = x[indices]
        y = y[indices]
        z = z[indices]

        # Plot the first lens surface
        xyz = self.transform.apply_transform_(
            torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1))  # Lens space to world space
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)

        # Plot the second lens surface
        xyz = self.transform.apply_transform_(
            torch.cat((-x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1))  # Lens space to world space
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kwargs)

    def sample_points_on_lens(self, nb_points, device='cpu'):
        raise NotImplementedError("Not implemented yet.")

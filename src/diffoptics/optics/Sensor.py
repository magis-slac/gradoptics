import torch

from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.transforms.Transforms import get_look_at_transform


class Sensor(BaseOptics):

    def __init__(self, resolution=(9600, 9600), pixel_size=(3.76e-6, 3.76e-6), position=(-0.057499999999999996, 0, 0),
                 poisson_noise_mean=2, quantum_efficiency=0.8, viewing_direction=torch.tensor([1., 0., 0.]),
                 up=torch.tensor([0., 0., 1.])):
        """

        :param resolution: Image processing convention: origin in the upper left corner, horizontal x axis and vertical
                           y axis (tuple)
        :param pixel_size: Image processing convention: origin in the upper left corner, horizontal x axis and vertical
                           y axis
        :param position: Position of the lens (list)
        :param poisson_noise_mean: Mean readout noise (float)
        :param quantum_efficiency: Quantum efficiency (float)
        :param viewing_direction: Viewing direction of the sensor (torch.tensor)
        :param up: A vector that orients the sensor with respect to the viewing direction (torch.tensor).
                   For example, if up=torch.tensor([0., 0., 1.]) the top of the camera will point upwards.
                   If up=torch.tensor([0., 0., -1.]), the top of the camera will point downwards.
        """
        super(Sensor, self).__init__()
        self.position = position
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.image = torch.zeros((resolution[0], resolution[1]))
        self.poisson_noise_mean = poisson_noise_mean
        self.quantum_efficiency = quantum_efficiency
        self.camera_to_world, self.world_to_camera = get_look_at_transform(viewing_direction, position, up=up)

    def get_ray_intersection(self, incident_rays):
        """
        Computes the time t at which the incident ray will intersect the sensor
        Note: Assume that the sensor is perpendicular to the optical axis
        :param incident_rays:
        :return:
        """
        origins = incident_rays.origins
        directions = incident_rays.directions
        t = (self.position[0] - origins[:, 0]) / directions[:, 0]
        y = origins[:, 1] + t * directions[:, 1]
        z = origins[:, 2] + t * directions[:, 2]

        condition = (t > 0) & (torch.abs(y - self.position[1]) <= self.resolution[0] / 2 * self.pixel_size[0]) & \
                    (torch.abs(z - self.position[2]) <= self.resolution[1] / 2 * self.pixel_size[1])
        t[~condition] = float('nan')
        return t

    def intersect(self, incident_rays, t, do_pixelize=True, quantum_efficiency=True):
        """
        @Todo
        :param incident_rays:
        :param t:
        :param do_pixelize:
        :param quantum_efficiency:
        :return:
        """

        origins = incident_rays.origins
        directions = incident_rays.directions
        hit_positions = origins + t.unsqueeze(1) * directions

        # World-space to camera-space
        hit_positions = torch.matmul(self.world_to_camera.to(hit_positions.device),
                                     torch.cat((hit_positions, torch.ones((hit_positions.shape[0], 1),
                                                                          device=hit_positions.device)), dim=1
                                               ).unsqueeze(-1))[:, :3, 0]

        # Camera space (origin in the center of the image, horizontal x axis pointing to the right and vertical y axis
        # pointing upwards) to python convention (origin in the upper left, vertical x axis and horizontal y axis)
        nb_horizontal_pixel_from_center = hit_positions[:, 0] / (self.pixel_size[1])
        nb_vertical_pixel_from_center = hit_positions[:, 1] / (self.pixel_size[0])
        hit_positions[:, 0] = -nb_vertical_pixel_from_center + self.resolution[1] // 2
        hit_positions[:, 1] = -nb_horizontal_pixel_from_center + self.resolution[0] // 2

        if do_pixelize:
            self.pixelize(hit_positions, quantum_efficiency=quantum_efficiency)

        return hit_positions, incident_rays.luminosities

    def pixelize(self, hit_positions, quantum_efficiency=True):
        """
        @Todo
        :param quantum_efficiency:
        :param hit_positions: batch of ...
        :return:
        """
        self.image = self.image.to(hit_positions.device)

        # Only keep the rays that make it to the sensor
        mask = (hit_positions[:, 0] < self.resolution[1]) & (hit_positions[:, 1] < self.resolution[0]) & \
               (hit_positions[:, 0] >= 0) & (hit_positions[:, 1] >= 0)
        hit_positions = hit_positions[mask]
        del mask

        if quantum_efficiency:  # Throw out some of the rays
            mask = torch.bernoulli(
                torch.zeros(hit_positions.shape[0], device=hit_positions.device) + self.quantum_efficiency,
                generator=None, out=None).type(torch.bool)
            hit_positions = hit_positions[mask]
            del mask

        # Update pixel values
        scale = max(self.resolution)
        indices = torch.floor(hit_positions[:, 0]).type(torch.int64) * scale + torch.floor(hit_positions[:, 1]).type(
            torch.int64)
        indices_and_counts = indices.unique(return_counts=True)
        tmp = torch.zeros(self.image.shape, device=self.image.device)
        for cnt in indices_and_counts[1].unique():
            ind = indices_and_counts[0][indices_and_counts[1] == cnt]
            tmp[ind // scale, ind % scale] += cnt

        self.image = self.image + tmp

    def readout(self, add_poisson_noise=True):
        tmp = self.image.clone()
        self.image = torch.zeros((self.resolution[0], self.resolution[1]), device=tmp.device)

        if add_poisson_noise:  # Add readout noise
            tmp = tmp + torch.poisson(self.image + self.poisson_noise_mean, generator=None)

        return tmp

    def plot(self, ax, s=0.1, color='grey', resolution_=100):
        """
        @Todo
        :param ax:
        :param s:
        :param color:
        :param resolution_:
        :return:
        """
        res = self.resolution[0] * self.pixel_size[0] / resolution_
        y = torch.arange(-self.resolution[0] * self.pixel_size[0] / 2 + self.position[1],
                         self.resolution[0] * self.pixel_size[0] / 2 + self.position[1],
                         res)
        z = torch.arange(-self.resolution[1] * self.pixel_size[1] / 2 + self.position[2],
                         self.resolution[1] * self.pixel_size[1] / 2 + self.position[2],
                         res)
        y, z = torch.meshgrid(y, z)
        x = torch.zeros(y.shape[0] * y.shape[1]) + self.position[0]
        ax.scatter(x, y.reshape(-1), z.reshape(-1), s=s, c=color)

    def plot_image(self, ax, img_height=750, cmap="Blues"):
        # @Todo
        raise NotImplementedError("Not implemented yet.")

    def sample_points_on_sensor(self, nb_points, device='cpu'):
        """
        Useful for backward ray tracing
        :return:
        """
        points = torch.zeros((nb_points, 3), device=device)
        points[:, 0] = self.position[0]
        points[:, 1] = torch.rand(nb_points, device=device) * (self.pixel_size[1] * self.resolution[1]) - (
                self.pixel_size[1] * self.resolution[1] / 2 - self.position[1])
        points[:, 2] = torch.rand(nb_points, device=device) * (self.pixel_size[0] * self.resolution[0]) - (
                self.pixel_size[0] * self.resolution[0] / 2 - self.position[2])
        return points

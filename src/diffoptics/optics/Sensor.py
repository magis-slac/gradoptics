import torch

from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.transforms.LookAtTransform import LookAtTransform

class Sensor(BaseOptics):

    def __init__(self, resolution=(9600, 9600), pixel_size=(3.76e-6, 3.76e-6), position=(-0.057499999999999996, 0, 0),
                 poisson_noise_mean=2, quantum_efficiency=0.8, viewing_direction=torch.tensor([1., 0., 0.]),
                 up=torch.tensor([0., 0., 1.]), psfs={}, psf_ratio=1):
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
        self.poisson_noise_mean = poisson_noise_mean
        self.quantum_efficiency = quantum_efficiency
        self.c2w = LookAtTransform(viewing_direction, position, up=up)
        self.add_psf = len(psfs.keys()) > 0
        self.psfs = psfs
        self.psf_ratio = psf_ratio
        self.resolution = resolution
        assert type(psf_ratio) == int

        if self.add_psf:
            self.depth_images = [torch.zeros((resolution[1] * psf_ratio, resolution[0] * psf_ratio, 1)) for _ in
                                 range(len(psfs['data'].keys()))]
            self.psf_depth_bounds = list(psfs['data'].keys())
        else:
            self.depth_images = [torch.zeros((resolution[1] * psf_ratio, resolution[0] * psf_ratio, 1))]

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
        hit_positions = torch.matmul(self.c2w.inverse_transform.type(hit_positions.dtype).to(hit_positions.device),
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

            if self.add_psf:

                nb_processed_rays = 0
                for depth_id, (low_bounds, high_bounds) in enumerate(self.psf_depth_bounds):
                    # Get the mask for the rays at the depths in [low_bounds, high_bounds]
                    mask = (incident_rays.meta['depth'] < high_bounds) & (incident_rays.meta['depth'] > low_bounds)
                    hit_positions_psf = hit_positions[mask] * self.psf_ratio
                    self.pixelize(depth_id, hit_positions_psf, quantum_efficiency=quantum_efficiency)
                    nb_processed_rays += mask.sum()

                if not (nb_processed_rays == hit_positions.shape[0]):
                    raise Exception(
                        "Some rays were not processed: their origins' depths were not included in the psfs.")

            else:
                self.pixelize(0, hit_positions, quantum_efficiency=quantum_efficiency)

        return hit_positions, incident_rays.luminosities, incident_rays.meta

    def pixelize(self, depth_id, hit_positions, quantum_efficiency=True):
        """
        @Todo
        :param depth_id:
        :param quantum_efficiency:
        :param hit_positions: batch of ...
        :return:
        """
        self.depth_images[depth_id] = self.depth_images[depth_id].to(hit_positions.device)

        # Only keep the rays that make it to the sensor
        mask = (hit_positions[:, 0] < (self.resolution[1] * self.psf_ratio)) & (
                    hit_positions[:, 1] < (self.resolution[0] * self.psf_ratio)) & \
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
        indices = torch.floor(hit_positions[:, 1]).type(torch.int64) * (self.resolution[1] * self.psf_ratio) + \
                  torch.floor(hit_positions[:, 0]).type(torch.int64)
        indices_and_counts = indices.unique(return_counts=True)
        tmp = torch.zeros(self.depth_images[depth_id].shape, device=self.depth_images[depth_id].device)
        for cnt in indices_and_counts[1].unique():
            ind = indices_and_counts[0][indices_and_counts[1] == cnt]
            tmp[ind % (self.resolution[1] * self.psf_ratio), ind // (self.resolution[1] * self.psf_ratio)] += cnt

        self.depth_images[depth_id] = self.depth_images[depth_id] + tmp

    def readout(self, add_poisson_noise=True, destructive_readout=True):

        # If psfs were specified
        if self.add_psf:
            # Apply psf to each depth_image
            for i, depth in enumerate(self.psfs['data'].keys()):
                for key in self.psfs['data'][depth].keys():
                    self.depth_images[i][key[0]:key[1], :, 0] = self.psfs['data'][depth][key](
                        self.depth_images[i][key[0]:key[1], :, 0])

        # Summing all the depth images & rebinning
        image = (torch.nn.AvgPool2d(self.psf_ratio)(
            torch.cat(self.depth_images, dim=2).sum(dim=2).unsqueeze(0).unsqueeze(0)) * self.psf_ratio ** 2).squeeze(
            0).squeeze(0)

        if destructive_readout:
            # Reinitialize depth images
            for i in range(len(self.depth_images)):
                self.depth_images[i] *= 0

        # Add readout noise
        if add_poisson_noise:
            image = image + torch.poisson(torch.zeros_like(image) + self.poisson_noise_mean, generator=None)

        return image

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

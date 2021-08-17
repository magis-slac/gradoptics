import torch

from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.transforms.Transforms import get_look_at_transform


class Sensor(BaseOptics):

    def __init__(self, resolution=(9600, 9600), pixel_size=(3.76e-6, 3.76e-6), position=(-0.057499999999999996, 0, 0),
                 poisson_noise_mean=2, quantum_efficiency=0.8, viewing_direction=torch.tensor([1., 0., 0.]),
                 up=torch.tensor([0., 0., 1.])):
        super(Sensor, self).__init__()
        self.position = position
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.image = torch.zeros((resolution[0], resolution[1]))
        self.poisson_noise_mean = poisson_noise_mean
        self.quantum_efficiency = quantum_efficiency
        self.camera_to_world, self.world_to_camera = get_look_at_transform(viewing_direction, torch.tensor(position),
                                                                           up=up)

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
        hit_position = torch.empty((origins.shape[0], 3), device=incident_rays.device)
        hit_position[:, 0] = origins[:, 0] + t * directions[:, 0]
        hit_position[:, 1] = origins[:, 1] + t * directions[:, 1]
        hit_position[:, 2] = origins[:, 2] + t * directions[:, 2]

        if do_pixelize:
            self.pixelize(hit_position, quantum_efficiency=quantum_efficiency)

        return hit_position, incident_rays.luminosities

    def pixelize(self, hit_positions, quantum_efficiency=True):
        """
        @Todo
        :param quantum_efficiency:
        :param hit_positions: batch of ...
        :return:
        """
        self.image = self.image.to(hit_positions.device)

        hit_positions = torch.matmul(self.world_to_camera,
                                     torch.cat((hit_positions,
                                                torch.ones(hit_positions.shape[0], 1)), dim=1).unsqueeze(-1))[:, :3, 0]

        # Only keep the rays that make it to the sensor
        mask = (hit_positions[:, 0] < self.pixel_size[0] * (self.resolution[0] // 2 - 1)) & \
               (hit_positions[:, 1] < self.pixel_size[1] * (self.resolution[1] // 2 - 1)) & \
               (hit_positions[:, 0] >= - self.pixel_size[0] * (self.resolution[0] // 2)) & \
               (hit_positions[:, 1] >= - self.pixel_size[1] * (self.resolution[1] // 2))
        hit_positions = hit_positions[mask]
        del mask

        if quantum_efficiency:  # Throw out some of the rays
            mask = torch.bernoulli(
                torch.zeros(hit_positions.shape[0], device=hit_positions.device) + self.quantum_efficiency,
                generator=None, out=None).type(torch.bool)
            hit_positions = hit_positions[mask]
            del mask

        self.image = Pixelize.apply(self.image, hit_positions, self.position, self.pixel_size, self.resolution)

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


class Pixelize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, image, hit_positions, position, pixel_size, resolution):
        # ctx is a context object that can be used to stash information
        # for backward computation

        nb_horizontal_pixel_from_center = torch.round((hit_positions[:, 0]) / (pixel_size[0])).type(
            torch.int64)
        # Shift because indexing starts from the bottom left corner
        index_x = nb_horizontal_pixel_from_center + resolution[0] // 2
        nb_vertical_pixel_from_center = torch.round((hit_positions[:, 1]) / (pixel_size[1])).type(
            torch.int64)
        # Shift because indexing starts from the bottom left corner
        index_y = nb_vertical_pixel_from_center + resolution[1] // 2

        # Update pixel values
        scale = max(resolution)
        indices = index_x * scale + index_y
        indices_and_counts = indices.unique(return_counts=True)
        tmp = torch.zeros(image.shape, device=image.device)
        for cnt in indices_and_counts[1].unique():
            ind = indices_and_counts[0][indices_and_counts[1] == cnt]
            tmp[ind // scale, ind % scale] += cnt

        ctx.indices = indices
        ctx.op = tmp
        ctx.device = image.device
        return image + tmp

    @staticmethod
    def backward(ctx, grad_output):
        gx = torch.tensor([[[[1., 0., -1.],
                             [2., 0., -2.],
                             [1., 0., -1.]]]], device=ctx.device)

        gy = torch.tensor([[[[1., 2., 1.],
                             [0., 0., 0.],
                             [-1., -2., -1.]]]], device=ctx.device)

        # x in 3d disappears in 2d
        # y in 3d -> columns in image => second index (not flipped, just shifted)
        # z in 3d -> rows in image => first index (flipped and shifted)
        grad_y = torch.nn.functional.conv2d(ctx.op.unsqueeze(0).unsqueeze(0), gx, padding=1).squeeze(0).squeeze(0)
        grad_z = torch.nn.functional.conv2d(ctx.op.unsqueeze(0).unsqueeze(0), gy, padding=1).squeeze(0).squeeze(0)

        grad = torch.zeros((ctx.indices.shape[0], 3), device=ctx.device)
        grad[:, 1] = grad_output[ctx.indices // 9600, ctx.indices % 9600] * (
            grad_y[ctx.indices // 9600, ctx.indices % 9600])
        grad[:, 2] = grad_output[ctx.indices // 9600, ctx.indices % 9600] * (
            grad_z[ctx.indices // 9600, ctx.indices % 9600])

        return None, grad, None, None, None  # the first None can be replaced by torch.ones(image.shape)

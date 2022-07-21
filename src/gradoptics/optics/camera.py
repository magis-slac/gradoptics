from gradoptics.optics.base_optics import BaseOptics
from gradoptics.optics.ray import Rays
import torch


class Camera(BaseOptics):
    """
    Models a camera by compounding multiple optical elements.
    """

    def __init__(self, lens, sensor, intermediate_objects=[]):
        """
        :param lens: External lens - interface with the world (:py:class:`~gradoptics.optics.lens.Lens`)
        :param sensor: Sensor (:py:class:`~gradoptics.optics.sensor.Sensor`)
        :param intermediate_objects: list of optical elements between the external lens and the sensor
                                     (:obj:`list` of :py:class:`~gradoptics.optics.base_optics.BaseOptics`)
        """
        super(Camera, self).__init__()
        self.lens = lens
        self.sensor = sensor
        self.intermediate_objects = intermediate_objects

    def get_ray_intersection(self, incident_rays):
        return self.lens.get_ray_intersection(incident_rays)

    def intersect(self, incident_rays, t):
        rays = self.lens.intersect(incident_rays, t)
        for obj in self.intermediate_objects:
            t = obj.get_ray_intersection(rays)
            rays = obj.intersect(rays, t)
        t = self.sensor.get_ray_intersection(rays)
        self.sensor.intersect(rays, t, do_pixelize=True, quantum_efficiency=True)

        # Returns nan rays
        origins = torch.zeros_like(incident_rays.origins) + float('nan')
        directions = torch.zeros_like(incident_rays.directions) + float('nan')
        luminosities = (torch.zeros_like(incident_rays.luminosities) + float('nan'))
        meta = {}
        for key in incident_rays.meta.keys():
            meta[key] = torch.zeros_like(incident_rays.meta[key]) + float('nan')

        mask = torch.zeros(origins.shape[0], dtype=torch.bool, device=origins.device)  # No rays reflected or refracted
        return Rays(origins, directions, luminosities=luminosities, device=incident_rays.device, meta=meta), mask

    def plot(self, ax):
        self.lens.plot(ax)
        for obj in self.intermediate_objects:
            obj.plot(ax)
        self.sensor.plot(ax)

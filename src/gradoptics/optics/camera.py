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
        rays, _ = self.lens.intersect(incident_rays, t)
        for obj in self.intermediate_objects:
            t = obj.get_ray_intersection(rays)
            cond = ~torch.isnan(t)
            rays, _ = obj.intersect(rays[cond], t[cond])
        t = self.sensor.get_ray_intersection(rays)
        cond = ~torch.isnan(t)
        self.sensor.intersect(rays[cond], t[cond], do_pixelize=True, quantum_efficiency=True)

        mask = torch.zeros(incident_rays.origins.shape[0], dtype=torch.bool, device=incident_rays.origins.device)  # No rays reflected or refracted
        return None, mask # No rays reflected / refracted

    def plot(self, ax):
        self.lens.plot(ax)
        for obj in self.intermediate_objects:
            obj.plot(ax)
        self.sensor.plot(ax)

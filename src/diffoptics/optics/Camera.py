from diffoptics.optics.BaseOptics import BaseOptics
from diffoptics.optics.Ray import Rays
import torch


class Camera(BaseOptics):

    def __init__(self, lens, sensor, intermediate_objects=[]):
        """
        :param lens: external lens - interface with the world
        :param sensor: sensor
        :param intermediate_objects: list of optical elements (BaseOptics) between the external lens and sensor
        """
        super(Camera, self).__init__()
        self.lens = lens
        self.sensor = sensor
        self.intermediate_objects = intermediate_objects

    def get_ray_intersection(self, incident_rays):
        """
        Computes the time t at which the incident ray will intersect the camera (i.e. the external lens)
        :param incident_rays:
        :return:
        """
        return self.lens.get_ray_intersection(incident_rays)

    def intersect(self, incident_rays, t):
        """
        Propagates the incident rays to the sensor.
        :param incident_rays:
        :param t:
        :return: nan rays
        """
        rays = self.lens.intersect(incident_rays, t)
        for obj in self.intermediate_objects:
            t = obj.get_ray_intersection(rays)
            rays = obj.intersect(rays, t)
        t = self.sensor.get_ray_intersection(rays)
        self.sensor.intersect(rays, t, do_pixelize=True, quantum_efficiency=True)

        # Return nan rays
        origins = torch.zeros_like(incident_rays.origins) + float('nan')
        directions = torch.zeros_like(incident_rays.directions) + float('nan')
        luminosities = (torch.zeros_like(incident_rays.luminosities) + float(
            'nan')) if incident_rays.luminosities is not None else None
        meta = {}
        for key in incident_rays.meta.keys():
            meta[key] = torch.zeros_like(incident_rays.meta[key]) + float('nan')
        return Rays(origins, directions, luminosities=luminosities, device=incident_rays.device, meta=meta)

    def plot(self, ax):
        self.lens.plot(ax)
        for obj in self.intermediate_objects:
            obj.plot(ax)
        self.sensor.plot(ax)

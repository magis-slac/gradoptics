from diffoptics.optics import BaseOptics
from diffoptics.optics.Sensor import Sensor
from diffoptics.light_sources.BaseLightSource import BaseLightSource

class Scene:

    def __init__(self, sensor: Sensor, light_source: BaseLightSource):
        self.sensor = sensor
        self.light_source = light_source
        self.objects = []

    def add_object(self, obj : BaseOptics, is_lens : bool):
        self.objects.append([obj, is_lens])

    def plot(self, ax):
        self.sensor.plot(ax)
        for obj, _ in self.objects:
            obj.plot(ax)

from diffoptics.optics import BaseOptics


class Scene:

    def __init__(self, sensor):
        self.sensor = sensor
        self.objects = []

    def add_object(self, obj : BaseOptics, is_lens : bool):
        self.objects.append([obj, is_lens])

    def plot(self, ax):
        self.sensor.plot(ax)
        for obj, _ in self.objects:
            obj.plot(ax)

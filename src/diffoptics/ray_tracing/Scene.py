
class Scene:
    """
    Models a scene made up of multiple optical elements
    """

    def __init__(self, sensor, light_source):
        """
        :param sensor: A sensor (:py:class:`~diffoptics.optics.Sensor.Sensor`)
        :param light_source: A light source (:py:class:`~diffoptics.light_sources.BaseLightSource.BaseLightSource`)
        """
        self.sensor = sensor
        self.light_source = light_source
        self.objects = []

    def add_object(self, obj, is_lens):
        """
        Adds an optical element to the scene

        :param obj: An optical element (:py:class:`~diffoptics.optics.BaseOptics.BaseOptics`)
        :param is_lens: Whether the optical element is a lens or no (:obj:`bool`)
        """
        self.objects.append([obj, is_lens])

    def plot(self, ax):
        """
        Plot the scene on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        self.sensor.plot(ax)
        self.light_source.plot(ax)
        for obj, _ in self.objects:
            obj.plot(ax)

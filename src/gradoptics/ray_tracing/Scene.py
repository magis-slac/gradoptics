
class Scene:
    """
    Models a scene made up of multiple optical elements
    """

    def __init__(self, light_source):
        """
        :param light_source: A light source (:py:class:`~gradoptics.light_sources.BaseLightSource.BaseLightSource`)
        """
        self.light_source = light_source
        self.objects = []

    def add_object(self, obj):
        """
        Adds an optical element to the scene

        :param obj: An optical element (:py:class:`~gradoptics.optics.BaseOptics.BaseOptics`)
        """
        self.objects.append(obj)

    def plot(self, ax):
        """
        Plot the scene on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        self.light_source.plot(ax)
        for obj in self.objects:
            obj.plot(ax)

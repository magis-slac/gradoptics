import abc  # Abstract Base Classes


class BaseOptics(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_ray_intersection(self, incident_rays, eps=1e-15):
        return NotImplemented

    @abc.abstractmethod
    def intersect(self, incident_rays, t):
        return NotImplemented

    @abc.abstractmethod
    def plot(self, ax):
        return NotImplemented

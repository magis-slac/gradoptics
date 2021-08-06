import abc  # Abstract Base Classes


class BaseLightSource(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def sample_rays(self, nb_rays, device='cpu'):
        return NotImplemented

    @abc.abstractmethod
    def plot(self, ax):
        return NotImplemented

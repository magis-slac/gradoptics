import abc  # Abstract Base Classes


class BaseDistribution(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample(self, nb_points, device='cpu'):
        raise NotImplemented

    @abc.abstractmethod
    def pdf(self, x):
        raise NotImplemented

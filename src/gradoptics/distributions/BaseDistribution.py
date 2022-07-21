import abc  # Abstract Base Classes


class BaseDistribution(abc.ABC):
    """
    Base class for probability distributions.
    """

    @abc.abstractmethod
    def sample(self, nb_points, device='cpu'):
        """
        Samples from the distribution

        :param nb_points: Number of points to sample (:obj:`int`)
        :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

        :return: Sampled points (:obj:`torch.tensor`)
        """
        raise NotImplemented

    @abc.abstractmethod
    def pdf(self, x):
        """
        Returns the pdf function evaluated at ``x``

        .. warning::
           The pdf may be unnormalized

        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)

        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        raise NotImplemented

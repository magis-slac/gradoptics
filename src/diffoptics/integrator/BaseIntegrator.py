import abc


class BaseIntegrator(abc.ABC):
    """
    Base class for integrators.
    """

    @abc.abstractmethod
    def compute_integral(self, incident_rays, pdf, t_min, t_max):
        """
        Computes the line integral between ``t_min`` and ``t_max`` of the incident rays through the density ``pdf``.

        :param incident_rays: The rays for which the line integrals should be computed
                              (:py:class:`~diffoptics.optics.Ray.Rays`)
        :param pdf: The pdf for which line integrals should be computed
        :param t_min: Lower integration bounds (:obj:`torch.tensor`)
        :param t_max: Higher integration bounds (:obj:`torch.tensor`)

        :return: Computed lines integrals (:obj:`torch.tensor`)
        """
        return NotImplemented

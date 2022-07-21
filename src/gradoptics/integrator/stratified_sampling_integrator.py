import torch
from gradoptics.integrator.base_integrator import BaseIntegrator


class StratifiedSamplingIntegrator(BaseIntegrator):
    """
    Computes line integrals using stratified sampling
    """

    def __init__(self, nb_mc_steps):
        """
        :param nb_mc_steps: Number of Monte Carlo integration steps used for approximating the integral (:obj:`int`)
        """
        self.nb_mc_steps = nb_mc_steps

    def compute_integral(self, incident_rays, pdf, t_min, t_max):

        # Interpolate t between t_min and t_max
        t = torch.linspace(0, 1, self.nb_mc_steps + 1, device=incident_rays.origins.device)
        t = (t_max - t_min).expand((self.nb_mc_steps + 1, t_max.shape[0])).T * t + \
            t_min.expand((self.nb_mc_steps + 1, t_max.shape[0])).T  # [nb_rays, nb_MC_steps]

        # Perturb sampling along each ray
        mid = (t[:, :-1] + t[:, 1:]) / 2.
        lower = torch.cat((t[:, :1], mid), -1)
        upper = torch.cat((mid, t[:, -1:]), -1)
        u = torch.rand(t.shape, device=incident_rays.origins.device)
        t = lower + (upper - lower) * u  # [nb_rays, nb_MC_steps]
        delta = t[:, 1:] - t[:, :-1]

        # 3d positions at the different times t
        x = incident_rays.origins.expand(self.nb_mc_steps, -1, -1).transpose(0, 1) + mid.unsqueeze(
            -1) * incident_rays.directions.expand(self.nb_mc_steps, -1, -1).transpose(0, 1)

        densities = pdf(x.reshape(-1, 3)).reshape((x.shape[:2]))

        return (densities * delta).sum(dim=1)

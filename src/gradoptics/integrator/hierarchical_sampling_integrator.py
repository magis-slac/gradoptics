import torch
from gradoptics.integrator.base_integrator import BaseIntegrator

class HierarchicalSamplingIntegrator(BaseIntegrator):
    """
    Computes line integrals using hierarchical sampling
    """

    def __init__(self, nb_mc_steps, nb_importance_samples, stratify=True, 
                 with_color=False, with_var=False):
        """
        :param nb_mc_steps: Number of Monte Carlo integration steps used for approximating the integral (:obj:`int`)
        """
        self.nb_mc_steps = nb_mc_steps
        self.nb_importance_samples = nb_importance_samples
        self.stratify = stratify
        self.with_color = with_color
        self.with_var = with_var
        
    def sample_pdf(self, bins, weights, n_samples, det=False):
        # This implementation is from NeRF
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # Take uniform samples
        if det:
            u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=weights.device)
            u = u.expand(list(cdf.shape[:-1]) + [n_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def compute_integral(self, incident_rays, pdf, t_min, t_max):
        t_vals = torch.linspace(0., 1., steps=self.nb_mc_steps + 1, device=incident_rays.origins.device)
        z_vals = t_min[:, None] * (1.-t_vals[None, :]) + t_max[:, None] * (t_vals[None, :])

        z_vals = z_vals.expand([incident_rays.origins.shape[0], self.nb_mc_steps+1])

        if self.stratify > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=incident_rays.origins.device)

            z_vals = lower + (upper - lower) * t_rand

        pts = incident_rays.origins[...,None,:] + incident_rays.directions[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        x_vals_mid = incident_rays.origins.expand(self.nb_mc_steps, -1, -1).transpose(0, 1) + z_vals_mid.unsqueeze(
                -1) * incident_rays.directions.expand(self.nb_mc_steps, -1, -1).transpose(0, 1)
        
        if self.nb_importance_samples > 0:
            deltas = z_vals[:, 1:] - z_vals[:, :-1]
            weights = pdf(x_vals_mid.reshape(-1, 3)).reshape((x_vals_mid.shape[:2]))*deltas
        
            z_imp = self.sample_pdf(z_vals_mid, 
                                    weights[...,1:], self.nb_importance_samples, det=(self.stratify==0.))
            z_imp = z_imp.detach()

            z_vals = torch.cat((z_imp, z_vals), dim=-1)
            z_vals, index = torch.sort(z_vals, dim=-1)
            
        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # 3d positions at the different times t
        x = incident_rays.origins.expand(z_vals_mid.shape[-1], -1, -1).transpose(0, 1) + z_vals_mid.unsqueeze(
            -1) * incident_rays.directions.expand(z_vals_mid.shape[-1], -1, -1).transpose(0, 1)

        if self.with_color:
            directions = incident_rays.directions.expand(z_vals_mid.shape[-1], -1, -1).transpose(0, 1)
            densities = pdf(x.reshape(-1, 3), directions.reshape(-1, 3)).reshape((x.shape[:2]))
        else:
            densities = pdf(x.reshape(-1, 3)).reshape((x.shape[:2]))

        if self.with_var:
            variances = pdf(x.reshape(-1, 3), return_var=True).reshape((x.shape[:2]))
            return (densities * deltas).sum(dim=1), (variances * (deltas**2)).sum(dim=1)
        else:
            return (densities * deltas).sum(dim=1)
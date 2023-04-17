import torch
import math

from gradoptics.distributions.base_distribution import BaseDistribution
from gradoptics.distributions.gaussian_distribution import GaussianDistribution
from gradoptics.inference.rejection_sampling import rejection_sampling


class AtomCloud(BaseDistribution):
    """
    Atom cloud with a sine-wave density modulation fringe pattern. Form and defaults from studies for MAGIS experiment.
    """

    def __init__(self, n=int(1e6), f=2, position=[0.31, 0., 0.], w0=0.0005, h_bar=1.0546 * 1e-34, m=1.44 * 1e-25,
                 t_final_bs=3., t_extra=0.1, port_bvz=.15, k_fringe=1 / (0.00003*2), a_quad=1e-12, phi=0.1,
                 phi2=math.pi / 2, proposal_distribution=GaussianDistribution(mean=0.0, std=0.0005)):
        """
        :param n: Number of atoms (:obj:`int`)
        :param position: Position of the center of the atom cloud [m] (:obj:`list`)
        :param phi: Phase of the sine-wave density modulation fringe pattern (:obj:`float`)
        :param w0: Beam width [m]. Roughly standard deviation of the atom cloud. (:obj:`float`)
        :param h_bar: Planck's constant, [kg * m^2 / s] (:obj:`float`)
        :param m: Strontium atom mass [kg] (:obj:`float`)
        :param t_final_bs: Time until final beam splitter [s] (:obj:`float`)
        :param k_fringe: Spatial frequency of fringe [1 / m] (:obj:`float`)
        :param a_quad: Magnitude of quadratic term (:obj:`float`)
        :param proposal_distribution: Proposal distribution used in rejection sampling for sampling from the
                                      unnormalized cloud distribution. Following units, mean, std in [m]
                                      (:py:class:`~gradoptics.distributions.base_distribution.BaseDistribution`)
        """
        super().__init__()
        self.n = n
        self.f = f
        self.position = torch.tensor(position)
        self.w0 = w0
        self.h_bar = h_bar
        self.m = m
        self.tFinalBS = t_final_bs
        self.tExtra = t_extra
        self.portBvz = port_bvz
        self.kFringe = k_fringe
        self.aQuad = a_quad
        self.phi2 = phi2
        self.phi = phi
        self.proposal_distribution = proposal_distribution

        # Define a sampler to sample from the cloud density (using rejection sampling)
        self.density_samplers = [lambda pdf, nb_point, device: rejection_sampling(pdf, nb_point, proposal_distribution,
                                                                                  m=None, device=device) for _ in
                                 range(3)]

    def marginal_cloud_density_x(self, x):
        """
        Returns the marginal pdf function along the x axis, evaluated at ``x``

        .. warning::
           The pdf is unnormalized

        :param x: Value where the pdf should be evaluated , in meters (:obj:`torch.tensor`)

        :return: The marginal pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        x = x.clone().type(torch.float64)

        dnr = ((-1. * ((2. * self.m * self.w0 * self.w0) + (1j * self.tFinalBS * self.h_bar)) ** 1.) ** .5)
        nrc = 1j * ((2. / math.pi) ** (1. / 4.)) * ((self.m * self.w0) ** (1. / 2.))
        psi1_x = nrc * torch.exp(-1 * self.m * (x ** 2) / (
                (4. * self.m * self.w0 * self.w0) + (2j * self.tFinalBS * self.h_bar))) / dnr
        density = torch.abs(psi1_x) ** 2
        return density

    def marginal_cloud_density_y(self, y):
        """
        Returns the marginal pdf function along the y axis, evaluated at ``y``

        .. warning::
           The pdf is unnormalized

        :param y: Value where the pdf should be evaluated , in meters (:obj:`torch.tensor`)

        :return: The marginal pdf function evaluated at ``y`` (:obj:`torch.tensor`)
        """

        y = y.clone().type(torch.float64)

        dnr = ((-1. * ((2. * self.m * self.w0 * self.w0) + (1j * self.tFinalBS * self.h_bar)) ** 1.) ** .5)
        nrc = 1j * ((2. / math.pi) ** (1. / 4.)) * ((self.m * self.w0) ** (1. / 2.))
        psi1 = nrc * ((1. / (2 ** .5)) + (torch.exp(
            (1j * self.phi) + (1j * ((self.kFringe * y) + (self.aQuad * self.kFringe * self.kFringe * y * y)))) / (
                                                  2 ** .5)))
        psi1_y = psi1 * torch.exp(-1 * self.m * (y ** 2) / (
                (4. * self.m * self.w0 * self.w0) + (2j * self.tFinalBS * self.h_bar))) / dnr
        density = torch.abs(psi1_y) ** 2
        return density

    def marginal_cloud_density_z(self, z):
        """
        Returns the marginal pdf function along the z axis, evaluated at ``z``

        .. warning::
           The pdf is unnormalized

        :param z: Value where the pdf should be evaluated , in meters (:obj:`torch.tensor`)

        :return: The marginal pdf function evaluated at ``z`` (:obj:`torch.tensor`)
        """

        z = z.clone().type(torch.float64)

        dnr = ((-1. * ((2. * self.m * self.w0 * self.w0) + (1j * self.tFinalBS * self.h_bar)) ** 1.) ** .5)
        nrc = 1j * ((2. / math.pi) ** (1. / 4.)) * ((self.m * self.w0) ** (1. / 2.))
        psi1_z = nrc * torch.exp(-1 * self.m * (z ** 2) / (
                (4. * self.m * self.w0 * self.w0) + (2j * self.tFinalBS * self.h_bar))) / dnr
        density = torch.abs(psi1_z) ** 2
        return density

    def pdf(self, x):  # @Todo, refractor. x,y,z -> x
        """
        Returns the pdf function evaluated at ``x``

        .. warning::
           The pdf is unnormalized

        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)

        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        return self.marginal_cloud_density_x(x[:, 0] - self.position[0]) * \
               self.marginal_cloud_density_y(x[:, 1] - self.position[1]) * \
               self.marginal_cloud_density_z(x[:, 2] - self.position[2])

    def sample(self, nb_points, device='cpu'):
        atoms = torch.empty((nb_points, 3))
        # Sample the cloud in the first dimension (Gaussian)
        tmp = self.density_samplers[0](self.marginal_cloud_density_x, nb_points, device)
        atoms[:, 0] = tmp
        # Sample the cloud in the second dimension (Gaussian)
        tmp = self.density_samplers[1](self.marginal_cloud_density_y, nb_points, device)
        atoms[:, 1] = tmp
        # Sample the cloud in the third dimension (Fringes)
        tmp = self.density_samplers[2](self.marginal_cloud_density_z, nb_points, device)
        atoms[:, 2] = tmp

        # Translate the cloud to its expected position
        ray_origins = atoms + self.position
        del atoms
        return ray_origins

    def plot(self, ax, **kwargs):
        """
        Plots the center of the atom cloud on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        ax.scatter(self.position[0], self.position[1], self.position[2], **kwargs)
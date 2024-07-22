from numpy import double
import torch
import math

from gradoptics.distributions.base_distribution import BaseDistribution
from gradoptics.distributions.gaussian_distribution import GaussianDistribution
from gradoptics.inference.rejection_sampling import rejection_sampling


def zernkike(xv: torch.Tensor, j: int, Rscale:double=1.0):
    # see https://en.wikipedia.org/wiki/Zernike_polynomials
    X = xv[:, 0]
    Y = xv[:, 1]
    R = torch.sqrt(X**2+Y**2)
    THETA = torch.atan2(Y, X)
    match j:
        case 0:
            F = torch.ones_like(X)
        case 1:
            F = X
        case 2:
            F = Y
        case 3:
            # Oblique astigmatism
            F = 2.*X.mul(Y)
        case 4:
            # Defocus
            F = X**2+Y**2
        case 5:
            # Vertical astigmatism
            F = X**2-Y**2
        case 6:
            # Vertical trefoil 
            F = torch.mul(R**3, torch.sin(3.*THETA))
        case 7:
            # Vertical coma
            F = torch.mul(3.*R**3,torch.sin(3.*THETA))
        case 8:
            # Horizontal coma 
            F = torch.mul(3.*R**3,torch.cos(3.*THETA))
        case 9:
            # Oblique trefoil 
            F = torch.mul(R**3, torch.cos(3.*THETA))
        case 10:
            # Oblique quadrafoil 
            F = 2.*torch.mul(R**4, torch.sin(4.*THETA))
        case 11:
            # Oblique secondary astigmatism 
            F = 2.*torch.mul(4.*R**4-3.*R**2, torch.sin(2.*THETA))
        case 12:
            # Primary spherical
            F = 6.*R**4-6.*R**2 + torch.ones_like(R)
        case 13:
            # Vertical secondary astigmatism 
            F = 2.*torch.mul(4.*R**4-3.*R**2, torch.cos(2.*THETA))
        case 14:
            # Vertical quadrafoil 
            F = 2.*torch.mul(R**4, torch.cos(4.*THETA))
        case _:
            raise
    
    return F


class AtomCloud_Aberration(BaseDistribution):
    """
    Atom cloud with a sine-wave density modulation fringe pattern. Form and defaults from studies for MAGIS experiment.
    """

    def __init__(self, n=int(1e6), f=2, position=[0.31, 0., 0.], w0=0.0005, h_bar=1.0546 * 1e-34, m=1.44 * 1e-25,
                 t_final_bs=3., t_extra=0.1, port_bvz=.15, k_fringe=1 / (0.00003*2), a_quad=1e-12, phi=0.1,
                 phi2=math.pi / 2, proposal_distribution=GaussianDistribution(mean=0.0, std=0.0005),
                 aberrationpars=[0.0 * 10], device='cpu'
                 ):
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
        self.position = torch.tensor(position, dtype=torch.float64, device=device)
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
        self.aberrationpars = aberrationpars
        self.device = device

        # define constants
        self.dnr = ((-1. * ((2. * self.m * self.w0 * self.w0) + (1j * self.tFinalBS * self.h_bar)) ** 1.) ** .5)
        print(self.dnr)
        self.atomcloudwidth = abs(self.dnr / self.m)
        self.nrc = 1j * ((2. / math.pi) ** (1. / 4.)) * ((self.m * self.w0) ** (1. / 2.))

        # Define a sampler to sample from the cloud density (using rejection sampling)
        self.density_samplers = [lambda pdf, nb_point, device: rejection_sampling(pdf, nb_point, proposal_distribution,
                                                                                  m=None, device=device) for _ in
                                 range(3)]

    def cloud_density_gaus(self, xv): 
        xv2sum = torch.sum(xv**2, dim=1)
        psi = self.nrc * torch.exp(-1 * self.m * (xv2sum) / (
                (4. * self.m * self.w0 * self.w0) + (2j * self.tFinalBS * self.h_bar))) / self.dnr
        density_gauss = torch.abs(psi) ** 2

        return density_gauss

    def cloud_density_phase(self, xv):
        x = xv[:, 0]
        y = xv[:, 1]

        phi_eff = self.phi + self.aberr_phase_zernike(xv)
        #phi_eff = self.phi + self.aberr_phase(xv)

        psi_phase = 0.5 * (1.  \
                + (torch.exp( 1j *( phi_eff + self.kFringe * y + self.aQuad * self.kFringe * self.kFringe * y * y)))
                )
 
        return torch.abs(psi_phase)**2
    
    def aberr_phase(self, xv):
        x = xv[:, 0]
        y = xv[:, 1]

        p = self.aberrationpars
        phi_eff = p[0] + p[1]*x + p[2]*y \
            + p[3]*x*x + p[4]*x*y + p[5]*y*y \
            + p[6]*x**3 + p[7]*x**2*y + p[8]*x*y**2 + p[9]*y**3
        
        return phi_eff
    
    # separate into zernike
    def aberr_phase_zernike(self, xv):
        p = self.aberrationpars

        for i in range(len(p)):
            phi_eff += p[i] * zernkike(xv, i) 
        return phi_eff

    
    def pdf(self, x):  # @Todo, refractor. x,y,z -> x
        """
        Returns the pdf function evaluated at ``x``

        .. warning::
           The pdf is unnormalized

        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)

        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        xv = (x - self.position)
        return self.cloud_density_gaus(xv) * self.cloud_density_phase(xv)

    def sample(self, nb_points, device='cpu'):

        accepted_data = torch.tensor([], device=device)
        npass = 0
        batch_size = int(nb_points / 2)
        while accepted_data.shape[0] < nb_points:
            atoms = torch.randn((nb_points, 3), dtype=torch.float64, device=device) * self.atomcloudwidth
            prob = self.cloud_density_phase(atoms)
            throwrnd = torch.rand(nb_points, device=device)
            accept = throwrnd < prob
            tmp_accepted_data = atoms[accept]
            accepted_data = torch.cat((accepted_data, tmp_accepted_data))

            del atoms
            del throwrnd
            del tmp_accepted_data


        # Translate the cloud to its expected position
        ray_origins = accepted_data[:nb_points, :] + self.position
        return ray_origins

    def plot(self, ax, **kwargs):
        """
        Plots the center of the atom cloud on the provided axes.

        :param ax: 3d axes (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`)
        """
        ax.scatter(self.position[0].cpu(), self.position[1].cpu(), self.position[2].cpu(), **kwargs)
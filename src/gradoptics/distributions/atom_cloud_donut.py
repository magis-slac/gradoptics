import torch
import math

from gradoptics.distributions.base_distribution import BaseDistribution
from gradoptics.distributions.gaussian_distribution import GaussianDistribution
from gradoptics.inference.rejection_sampling import rejection_sampling

class AtomCloudDonut(BaseDistribution):
    """
    Atom cloud "donut" with a hole in the middle. Motivated by discussion with Jason Hogan, et al. and implemented by Sanha.

    2D Gaussian in the transverse plane with a central hole "blown away". Cylindrically symmetric. See atom_cloud.py for base 
    atom cloud definition.
    """

    def __init__(self, n=int(1e6), position=[0.31, 0., 0.], r1=0.0005, r2=0.0002, p=2., sigma_z=0.0002,
                 transverse_proposal=None, longitudinal_proposal=None):
        """
        :param n: Number of atoms (:obj:`int`)
        :param position: Position of the center of the atom cloud [m] (:obj:`list`)
        :param r1: Atom cloud radius (Gaussian std) [m] (:obj:`float`)
        :param r2: Donut hole radius (Gaussian std) [m] (:obj:`float`)
        :param p: "Power" of the push-away beam (:obj:`float`)
        :param sigma_z: Atom cloud thickness (Gaussian std) in longitudinal direction [m] (:obj:`float`)
        :param transverse_proposal: Proposal distribution used in rejection sampling
                                      for sampling radial position from the transverse axis
                                      (:py:class:`~gradoptics.distributions.base_distribution.BaseDistribution`)
        :param longitudinal_proposal: Proposal distribution used in rejection sampling
                                      for sampling radial position from the transverse axis
                                      (:py:class:`~gradoptics.distributions.base_distribution.BaseDistribution`)                                    
        """
        super().__init__()
        self.n = n
        self.position = torch.tensor(position)
        self.r1 = r1
        self.r2 = r2
        self.p = p
        self.sigma_z = sigma_z

        if transverse_proposal:
            self.transverse_proposal = transverse_proposal
        else:
            self.transverse_proposal = GaussianDistribution(mean=0, std=r1)

        if longitudinal_proposal:
            self.longitudinal_proposal = longitudinal_proposal
        else:
            self.longitudinal_proposal = GaussianDistribution(mean=0, std=sigma_z)

        # Define a sampler to sample from the cloud density (using rejection sampling)
        # self.density_samplers[0] is the transverse, radial sampler
        # self.density_samplers[1] is the longitudinal sampler
        self.density_samplers = [lambda pdf, nb_point, device: rejection_sampling(pdf, nb_point, self.transverse_proposal,
                                                                                  m=None, device=device),
                                 lambda pdf, nb_point, device: rejection_sampling(pdf, nb_point, self.longitudinal_proposal,
                                                                                  m=None, device=device)
                                ]

    def marginal_cloud_density_r(self, r):
        """
        Returns the marginal pdf function along the radial axis, evaluated at ``r``
        .. warning::
           The pdf is unnormalized
        :param r: Value where the pdf should be evaluated , in meters (:obj:`torch.tensor`)
        :return: The marginal pdf function evaluated at ``r`` (:obj:`torch.tensor`)
        """
        r = r.clone().type(torch.float64)

        u = torch.exp(-.5 * (r / self.r1)**2)
        v = (1. - torch.exp(-.5 * (r / self.r2) **2)) ** self.p

        return u * v

    def marginal_cloud_density_phi(self, phi):
        """
        Returns the marginal pdf function along the azimuthal axis, evaluated at ``phi``
        .. warning::
           The pdf is unnormalized
        :param phi: Value where the pdf should be evaluated , in radians (:obj:`torch.tensor`)
        :return: The marginal pdf function evaluated at ``r`` (:obj:`torch.tensor`)
        """
        return 1 / (2 * math.pi)

    def marginal_cloud_density_z(self, z):
        """
        Returns the marginal pdf function along the longitudinal axis, evaluated at ``z``
        .. warning::
           The pdf is unnormalized
        :param z: Value where the pdf should be evaluated , in meters (:obj:`torch.tensor`)
        :return: The marginal pdf function evaluated at ``z`` (:obj:`torch.tensor`)
        """

        z = z.clone().type(torch.float64)

        return GaussianDistribution(mean=0, std=self.sigma_z).pdf(z)

    def pdf(self, x):  # @Todo, refractor. x,y,z -> x
        """
        Returns the pdf function evaluated at ``x``
        .. warning::
           The pdf is unnormalized
        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)
        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """
        r = torch.sqrt((x[:, 0] - self.position[0]) ** 2 + (x[:, 1] - self.position[1]) ** 2)
        phi = torch.atan2(x[:, 1] - self.position[1], x[:, 0] - self.position[0])
        return self.marginal_cloud_density_r(r) * \
               self.marginal_cloud_density_phi(phi) * \
               self.marginal_cloud_density_z(x[:, 2])

    def sample(self, nb_points, device='cpu'):
        atoms = torch.empty((nb_points, 3))

        # Sample in the transverse plane
        r_tmp = self.density_samplers[0](self.marginal_cloud_density_r, nb_points, device)
        phi_tmp = torch.rand(nb_points, device=device) * math.pi
        atoms[:, 0] = r_tmp * torch.cos(phi_tmp)
        atoms[:, 1] = r_tmp * torch.sin(phi_tmp)

        # Sample in the longitudinal axis
        tmp = self.density_samplers[1](self.marginal_cloud_density_z, nb_points, device)
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
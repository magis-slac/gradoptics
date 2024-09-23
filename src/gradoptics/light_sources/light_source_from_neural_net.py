import math
import torch

from gradoptics import Rays
from gradoptics.light_sources.base_light_source import BaseLightSource
from gradoptics.optics import batch_vector


class LightSourceFromNeuralNet(BaseLightSource):
    """
    Models a light source from a neural network.
    """

    def __init__(self, network, bounding_shape=None, 
                 rad=1., x_pos=0., y_pos=0., z_pos=0., logpdf=False,
                 network_var=None):
        """
        :param network: Neural network representation of density
        :param bounding_shape: A bounding shape that bounds the light source
                               (:py:class:`~gradoptics.optics.bounding_shape.BoundingShape`). Default is ``None``
        .. note::
             A bounding shape is required if this light source is used with backward ray tracing
        """
        self.network = network
        self.network_var = network_var
        self.bounding_shape = bounding_shape
        
        self.rad = rad
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        
        trans_mat = torch.eye(4)
        trans_mat[0][3] = -self.x_pos* 1/self.rad
        trans_mat[1][3] = -self.y_pos* 1/self.rad
        trans_mat[2][3] = -self.z_pos* 1/self.rad

        scale_mat = torch.eye(4)
        scale_mat[0][0] = 1/self.rad
        scale_mat[1][1] = 1/self.rad
        scale_mat[2][2] = 1/self.rad

        full_mat = torch.matmul(trans_mat, scale_mat)
        self.full_scale_mat = full_mat[:-1]
        
        self.inv_scale_mat = torch.inverse(full_mat)

        self.logpdf = logpdf

    def sample_rays(self, nb_rays, device='cpu', sample_in_2pi=False):
        pass

    def plot(self, ax, **kwargs):
        pass

    def pdf(self, x, viewdir=None, return_var=False):
        """
        Returns the pdf function of the distribution evaluated at ``x``
        .. warning::
           The pdf may be unnormalized
        :param x: Value where the pdf should be evaluated (:obj:`torch.tensor`)
        :return: The pdf function evaluated at ``x`` (:obj:`torch.tensor`)
        """

        if return_var:
            network_here = self.network_var
        else:
            network_here = self.network

        x_scale = torch.matmul(self.full_scale_mat.to(x.device).type(x.dtype), 
                               torch.cat((x, torch.ones((x.shape[0],1),
                                                        device=x.device, dtype=x.dtype)), dim=1)[:, :, None]).squeeze(dim=-1)
        if viewdir is not None:
            viewdir_scale = torch.matmul(self.full_scale_mat.to(viewdir.device).type(viewdir.dtype), 
                               torch.cat((viewdir, torch.zeros((viewdir.shape[0],1),
                                                        device=viewdir.device, dtype=viewdir.dtype)), dim=1)[:, :, None]).squeeze(dim=-1)
            
            pdf_val, color, coords = network_here(x_scale, viewdir_scale)
            if self.logpdf:
                self.pdf_val = torch.exp(pdf_val)
            else:
                self.pdf_val = pdf_val
            self.color = color
            
            if return_var:
                self.pdf_aux_var = coords
                return self.pdf_val*self.color**2
            else:
                self.pdf_aux = coords
                return self.pdf_val*self.color
        
        else:
            pdf_val, coords = network_here(x_scale)
            if self.logpdf:
                self.pdf_val = torch.exp(pdf_val)
            else:
                self.pdf_val = pdf_val
            
            if return_var:
                self.pdf_aux_var = coords
            else:
                self.pdf_aux = coords
            
            return self.pdf_val
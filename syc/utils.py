import numpy as np
import torch
import gradoptics as optics
from gradoptics.optics.ray import Rays
from gradoptics.ray_tracing.ray_tracing import trace_rays
from gradoptics.distributions.base_distribution import BaseDistribution

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.transform import Rotation as R

from knn import *
from voxel import create_voxel_grid

from numba import jit

def cos_theta(batch_vector1, batch_vector2, eps=1e-15):
    """
    Computes the cosine between the given batch of 3d vectors

    :param batch_vector1: A batch of 3d vectors (:obj:`torch.tensor`)
    :param batch_vector2: A batch of 3d vectors (:obj:`torch.tensor`)
    :param eps: Parameter used for numerical stability. Default is ``'1e-15'``

    :return: The cosine between ``batch_vector1`` and ``batch_vector2`` (:obj:`torch.tensor`)
    """
    norm_v1 = torch.sqrt(batch_vector1[:, 0] ** 2 + batch_vector1[:, 1] ** 2 + batch_vector1[:, 2] ** 2)
    norm_v2 = torch.sqrt(batch_vector2[:, 0] ** 2 + batch_vector2[:, 1] ** 2 + batch_vector2[:, 2] ** 2)
    return optics.dot_product(batch_vector1, batch_vector2) / (norm_v1 * norm_v2 + eps)

# get sensors in a scene

def get_sensors(scene):
    sensorslist = []
    for aobject in scene.objects:
        if isinstance(aobject, optics.Sensor):
            sensorslist.append(aobject)
    return sensorslist


#Calculate cluster positions and extents

# find the clusters and the maximum distance or ray from the centroid for each cluster
# find the efficiency 
def kwindows(data, dataraw, directionsnp):
    # find centroids
    windows = []

    # use scipy hierarchical clustering
    Z = linkage(data)
    labels = fcluster(Z, t=0.2, criterion='distance') - 1 # subtract 1 to start from index 0
    nclusters = np.max(labels) + 1
    k = nclusters
    # get centroids
    centroids = np.array([directionsnp[labels == i].mean(axis=0) for i in range(nclusters)])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    # distance from centroids
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

    maxdist = [1.1*np.max(distances[labels==i, i]) for i in range(k)]

    counts = [np.count_nonzero(labels==i) for  i in range(k)]

    # all data
    distances_all = np.linalg.norm(dataraw[:, np.newaxis] - centroids, axis=2)
    labels_all = np.argmin(distances_all, axis=1)
    
    counts_all = [np.count_nonzero(distances_all[labels_all==i, i] < maxdist[i]) for  i in range(k)]
    #counts_all = [np.count_nonzero(labels_all==i) for  i in range(k)]

    return centroids, maxdist, counts, counts_all


# custom weighted sampler

def custom_histogram(data:torch.Tensor, bin_edges:torch.Tensor) -> torch.Tensor:
    """
    Compute a histogram for 1D data using custom binning.

    Args:
        data (torch.Tensor): Input data (1D tensor).
        num_bins (int): Number of bins for the histogram.

    Returns:
        torch.Tensor: Histogram counts.
    """

    num_bins = bin_edges.shape[0]-1
    device = data.device
    # Initialize histogram counts
    hist_counts = torch.zeros(num_bins, dtype=torch.int64, device=device)

    # Iterate over data points and assign to bins
    for value in data:
        bin_index = torch.searchsorted(bin_edges, value, right=True) - 1
        hist_counts[bin_index] += 1

    return hist_counts

def compute_bin_indices_non_equidistant(input_points, bin_edges):
    """
    Compute bin indices for 3D points with non-equidistant bin edges.

    Args:
        input_points (torch.Tensor): Input tensor of shape (N, 3) containing 3D points.
        bin_edges (torch.Tensor): Tensor of shape (3, nbins+1) containing non-equidistant bin edges.

    Returns:
        torch.Tensor: Output tensor of shape (N, 3) containing bin indices for each point.
    """
    # Initialize an empty tensor to store bin indices
    bin_indices = torch.empty_like(input_points, dtype=torch.long)

    # Find the bin index for each point along each dimension
    for dim in range(3):
        bin_indices[:, dim] = torch.searchsorted(bin_edges[dim], input_points[:, dim]) - 1

    return bin_indices



def weighted_sampler(nsamples, centroids, maxdist, counts):
    nregions = centroids.shape[0]
    prob_per_region = counts / np.sum(counts)
    
    indices = np.random.choice(nregions, size=nsamples, p=prob_per_region)
    counts_all = [np.count_nonzero(indices==i) for i in range(nregions)]
    resultlist=[]
    for it in enumerate(counts_all):
        centroid_theta = np.expand_dims([np.arccos(centroids[it[0] ,2])], axis=1)
        centroid_phi = np.expand_dims([np.arctan2(centroids[it[0], 1], centroids[it[0], 0])], axis=1)
        nsamples = it[1]

        mincosth = np.cos(maxdist[it[0]])
        # sample from a cone with centroid as axis
        phiprime = np.random.uniform(-np.pi, np.pi, size=nsamples)
        #uniform in cos theta within cone
        thetaprime = np.arccos(np.random.uniform(mincosth, 1.0, size=nsamples))

        x = np.sin(thetaprime) * np.cos(phiprime)
        y = np.sin(thetaprime) * np.sin(phiprime)
        z = np.cos(thetaprime)

        locdirections = np.hstack((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]))

        # global
  
        # very slow
        globaltrans = R.from_euler('ZY', np.hstack((centroid_phi, centroid_theta))) 
        
        directions = globaltrans.apply(locdirections)
        resultlist.append(directions)
    res = np.vstack(resultlist)
    
    return res, counts_all

# backward ray tracer to use point cloud sources
# due to gaussian envelope, we don't need to use any integrator
def backward_ray_tracing(incident_rays: optics.Rays, scene: optics.Scene, point_cloud_light_source: optics.LightSourceFromDistribution, max_iterations=3, sigma=5.0e-6):
    intensities = torch.zeros(incident_rays.get_size(), dtype=torch.double, device=incident_rays.device)

    # Labelling the rays
    incident_rays.meta['track_idx'] = torch.linspace(0, incident_rays.get_size() - 1, incident_rays.get_size(),
                                                     dtype=torch.long, device=incident_rays.device)
    incident_rays.meta['radiance_scaling'] = torch.ones(incident_rays.get_size(), device=incident_rays.device)
    with torch.no_grad():
        for i in range(max_iterations):
            outgoing_rays, t, mask = trace_rays(incident_rays, scene)

            # Potential intersection with the light source
            t_min, t_max = point_cloud_light_source.bounding_shape.get_ray_intersection_(incident_rays)

            # Mask for the rays that hit the light source rather than the object found in trace_rays
            new_mask = (t_min < t) & (t_min < t_max)

            # Computing the intensities for the rays that have hit the light source
            if new_mask.sum() > 0:
                raysclose = incident_rays[new_mask]

                intensitysum = point_cloud_light_source.distribution.pdf_tracks(raysclose.origins, raysclose.directions)
                intensities[incident_rays.meta['track_idx'][new_mask]] = intensitysum.detach()


            # Rays that are still in the scene, and have not hit the light source
            incident_rays = outgoing_rays[mask & (~new_mask)]

    return intensities


class FlatSphereDistribution(BaseDistribution):
    def __init__(self, position=[0, 0, 0], radius=0.05, thickness=0.001):
        self.position = position
        self.radius = radius
        self.thickness = thickness
        self.area = 4/3 * np.pi * self.radius**3
    pass

    # sample disc
    def sample(self, nb_points, device='cpu'):
        costheta = -1.0 + 2.0 * torch.rand(nb_points, device=device)
        theta= torch.acos(costheta)
        phi = 2 * np.pi * torch.rand(nb_points, device=device)
        r = self.radius * torch.pow(torch.rand(nb_points, device=device), 1/3)
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * costheta
        del costheta
        del phi
        return torch.tensor(self.position, device=device) + optics.batch_vector(x, y, z)

    def pdf(self, x):
        dev = x.device
        r = torch.norm((x-torch.tensor(self.position, device=dev)), dim=1, keepdim=True)
        ones = torch.ones_like(r, dtype=x.dtype)
        sphereradius = self.radius * ones
        sphereinnerradius = (self.radius - self.thickness) * torch.ones_like(r)

        #intensity = torch.heaviside(sphereradius-r, ones) * torch.heaviside( sphereinnerradius-r, ones) / self.area
        intensity = torch.heaviside(sphereradius-r, ones) / self.area
        return intensity
    

    def plot(self, ax, **kwargs):
        ax.scatter(self.position[0], self.position[1], self.position[2], **kwargs)
        pass

class FlatSquareDistribution(BaseDistribution):
    def __init__(self, position=(0, 0, 0), width=0.002, thickness=0.0001):
        self.position = position
        self.width = width
        self.thickness = thickness
        self.area = width**2
        pass

    def sample(self, nb_points, device='cpu'):
        x = (-0.5 + torch.rand(nb_points, device=device)) * self.width
        y = (-0.5 + torch.rand(nb_points, device=device)) * self.width
        z = (-0.5 + torch.rand(nb_points, device=device)) * self.thickness
        return torch.tensor(self.position, device=device) + optics.batch_vector(x, y, z)
    
    def pdf(self, x):
        dev = x.device
        checkx = torch.le(torch.abs(x[:, 0]), self.width/2)
        checky = torch.le(torch.abs(x[:, 1]), self.width/2)
        checkz = torch.le(torch.abs(x[:, 2]), self.thickness/2)

        inplane = torch.logical_and(checkx, checky)
        invol = torch.logical_and(inplane, checkz)

        intensities = torch.zeros_like(invol)
        intensities[invol] = 1/self.area

        return intensities
    
    def plot(self, ax, **kwargs):
        dh = self.width/2
        dz = self.thickness/2
        xs = np.array([-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1] )
        ys = np.array([-1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1] )
        zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1] )

        xs = dh* np.outer(xs, xs)
        ys = dh* np.outer(ys, ys)
        zs = dz* np.outer(zs, zs)

        ax.plot_surface(xs, ys, zs,  color=(1.0, 0.7, 0.0, 0.5))
        pass

        

# following doesn't work with backward ray tracing unless a matching 
# bounding box is made
class FlatDiscDistribution(BaseDistribution):
    def __init__(self, position=[0, 0, 0], radius=0.05, thickness=0.001):
        self.position = position
        self.radius = radius
        self.thickness = thickness
        self.area = np.pi * self.radius**2
    pass

    # sample disc
    def sample(self, nb_points, device='cpu'):
        phi = 2 * np.pi * torch.rand(nb_points, device=device)
        r = self.radius * torch.sqrt(torch.rand(nb_points, device=device))
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = self.thickness * (-0.5 + torch.rand(nb_points, device=device))

        del phi
        del r
        return torch.tensor(self.position, device=device) + optics.batch_vector(x, y, z)

    def pdf(self, x):
        dev = x.device
        dxvec = x -torch.tensor(self.position, device=dev)
        r = torch.norm(dxvec[:, :2], dim=1)
        checkr = torch.le(r, self.radius)
        checkz = torch.le(torch.abs(dxvec[:, 2]), torch.tensor(0.5*self.thickness, device=dev))
        invol = torch.logical_and(checkr, checkz)
        intensities = torch.zeros_like(invol)
        intensities[invol] = 1/self.area
        return intensities
    

    def plot(self, ax, **kwargs):
        ax.scatter(self.position[0], self.position[1], self.position[2], **kwargs)
        pass

    # define bounding box for box type
from gradoptics.optics.bounding_shape import BoundingShape
class BoundingBox(BoundingShape):

    def __init__(self, xc=0.2, yc=0.0, zc=0.0, dx=0.002, dy=0.002, dz=0.0001):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.xc = xc
        self.yc = yc
        self.zc = zc

    def get_ray_intersection_(self, incident_rays, eps=1e-10):
        """
        @Todo
        :param incident_rays:
        :param eps:
        :return:
        """
        
        # Computes the intersection of the incident_ray with the cube
        origins = incident_rays.origins
        orig_directions = incident_rays.directions

        
        directions = orig_directions/torch.norm(orig_directions, dim=1, keepdim=True)

        # Now that we've rotated, we're axis aligned. Set up the faces of the cube
        fx0 = self.xc - self.dx/2 
        fx1 = self.xc + self.dx/2
        
        fy0 = self.yc - self.dy/2 
        fy1 = self.yc + self.dy/2
        
        fz0 = self.zc - self.dz/2 
        fz1 = self.zc + self.dz/2
        
        # Get the time of intersection with the planes corresponding to each face
        tx0 = ((fx0 - origins[:, 0]) / directions[:, 0])[:, None]
        tx1 = ((fx1 - origins[:, 0]) / directions[:, 0])[:, None]
        
        ty0 = ((fy0 - origins[:, 1]) / directions[:, 1])[:, None]
        ty1 = ((fy1 - origins[:, 1]) / directions[:, 1])[:, None]
        
        tz0 = ((fz0 - origins[:, 2]) / directions[:, 2])[:, None]
        tz1 = ((fz1 - origins[:, 2]) / directions[:, 2])[:, None]
        
        # And the corresponding points of intersection
        pointx0 = origins + tx0*directions
        pointx1 = origins + tx1*directions
        
        pointy0 = origins + ty0*directions
        pointy1 = origins + ty1*directions
        
        pointz0 = origins + tz0*directions
        pointz1 = origins + tz1*directions
        
        # Package everything up in a 6 face x n rays x 1 (3) tensor
        all_ts = torch.stack([tx0, tx1, 
                              ty0, ty1,
                              tz0, tz1])
            
        all_int = torch.stack([pointx0, pointx1, 
                              pointy0, pointy1, 
                              pointz0, pointz1])
        
        # Select the points that fall within the cube volume (with some tolerance)
        in_cube_mask = ((all_int[:, :, 0] >= fx0-eps) & (all_int[:, :, 0] <= fx1+eps) & 
                        (all_int[:, :, 1] >= fy0-eps) & (all_int[:, :, 1] <= fy1+eps) &
                        (all_int[:, :, 2] >= fz0-eps) & (all_int[:, :, 2] <= fz1+eps))

        # at least two hits..
        raywithvalidhits = torch.count_nonzero(in_cube_mask, dim=0)>=1
        maxval = torch.finfo(torch.float64).max
        minval =  torch.finfo(torch.float64).min
        
       
        intersected = (all_ts>0)
        t_min = torch.zeros(all_ts.shape[1], device=incident_rays.device, dtype=origins.dtype)
        t_max = torch.zeros(all_ts.shape[1], device=incident_rays.device, dtype=origins.dtype)
        
        all_ts[~in_cube_mask] = maxval
        t_min[raywithvalidhits] = torch.min(all_ts[:,raywithvalidhits, 0], dim=0)[0]
        t_min[~raywithvalidhits] = float('nan')

        all_ts[~in_cube_mask] = minval
        t_max[raywithvalidhits] = torch.max(all_ts[:,raywithvalidhits, 0], dim=0)[0]
        t_max[~raywithvalidhits] = float('nan')

        # Return first intersection and corresponding (rotated) origin/direction 
        return t_min, t_max 
    
    def get_ray_intersection(self, incident_rays, eps=1e-10):
        return self.get_ray_intersection_(incident_rays, eps=eps)[0]

    def intersect(self, incident_rays, t):
        origins = incident_rays.origins
        directions = incident_rays.directions

        # Update the origin of the incoming rays
        origins = origins + t.unsqueeze(1) * directions

        return Rays(origins,
                    directions,
                    luminosities=incident_rays.luminosities,
                    device=incident_rays.device,
                    meta=incident_rays.meta)
 
    def plot(self, ax, color='grey', alpha=0.4):
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi)

        x = np.cos(Phi)*np.sin(Theta)
        y = np.sin(Phi)*np.sin(Theta)
        z = np.cos(Theta)/np.sqrt(2)
        
        ax.plot_surface(x*self.width+self.xc, 
                        y*self.width+self.yc, 
                        z*self.width+self.zc, color=color, alpha=alpha)
import gradoptics as optics
from gradoptics.optics.ray import Rays
from gradoptics.optics import normalize_batch_vector
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from glob import glob

from utils import cos_theta, get_sensors, kwindows, weighted_sampler

import torch
from scipy.spatial.transform import Rotation as R
import pickle
from pathlib import Path

# Define an integrator
from gradoptics.integrator import HierarchicalSamplingIntegrator
from gradoptics import LightSourceFromNeuralNet
from models.parametrizedcloud import parametrizedcloud
import pathlib
import pandas as pd

def point_to_origin(cam_position):
    # Default is point along positive x
    current_dir = torch.tensor([1., 0., 0.]).type(cam_position.dtype)
    
    # Vector pointing towards the origin
    new_dir = -cam_position/torch.norm(cam_position)
    
    # Get a perpendicular axis, handling axis aligned cases
    if torch.allclose(torch.abs(new_dir), torch.abs(current_dir)):
        axis = torch.tensor([0., 0., 1.]).double()
    else:
        axis = torch.cross(current_dir, new_dir)
     
    # Normalize
    axis *= 1/torch.norm(axis)
    
    # Get rotation angle via dot product
    angle = torch.acos(torch.dot(current_dir,new_dir)/(torch.norm(current_dir)*torch.norm(new_dir)))
    
    # Get Euler angles from rotation vector using scipy (right handed coordinate system)
    theta_x, theta_y, theta_z = R.from_rotvec(axis*angle).as_euler('xyz')
    
    # Gradoptics rotations need left handed coordinate system -- flip sign on y
    return theta_x, -theta_y, theta_z

def calculate_focal_length(m, obj_distance):
    f =  np.array(obj_distance) / (1 / np.array(m) + 1)
    return f.astype(np.float32)


def setup_scene(scene, thetas=[], phis=[], rs=[], m=[], na=[1/1.4], mode='40k'):
    #Make sure the number of cameras matches between arguments
    assert len(thetas) == len(phis), "thetas and phis should have the same length"
    assert len(rs) == len(thetas), "rs and thetas should have the same length"
    assert len(m) == len(thetas), "m and thetas should have the same"
    
    n_cameras = len(thetas)
    
    # Given distance from object and magnification, calculate focal length for in focus object
    #m = 0.1
    f = calculate_focal_length(m, rs)
    
    # Numerical aperture (size of camera opening, f-number)
    #na = 1/1.4
    if mode=='40k':
        res=(200, 200)
        pixsize = (2.4e-6, 2.4e-6)
    elif mode=='400':
        res=(20, 20)
        pixsize = (2.4e-5, 2.4e-5)
    elif mode=='40klarge':
        res=(200, 200)
        pixsize = (4.8e-6, 4.8e-6)
    elif mode=='400large':
        res=(20, 20)
        pixsize = (4.8e-5, 4.8e-5)
    else:
        print('wrong mode!')
        exit(1)
    
    # Loop over cameras to add to scene
    for i_cam in range(n_cameras):
        # Avoid singular point with slight offset
        if thetas[i_cam] == 0:
            thetas[i_cam] = 1e-6
            
        # Get cartesian coordinates from spherical
        x_cam = rs[i_cam]*np.sin(thetas[i_cam])*np.cos(phis[i_cam])
        y_cam = rs[i_cam]*np.sin(thetas[i_cam])*np.sin(phis[i_cam])
        z_cam = rs[i_cam]*np.cos(thetas[i_cam])
        
        cam_pos = torch.tensor([x_cam, y_cam, z_cam])

        # Get orientation to point at origin and apply to lens
        angles = point_to_origin(cam_pos)
        transform = optics.simple_transform.SimpleTransform(*angles, cam_pos)
        lens = optics.PerfectLens(f=f[i_cam], m=m[i_cam], na=na[i_cam],
                                  position = cam_pos,
                                  transform = transform)

        # Sensor position from lensmakers equation, rotated to match lens
        rel_position = torch.tensor([-f[i_cam] * (1 + m[i_cam]), 0, 0]).float()                       
        rot_position = torch.matmul(transform.transform.float(), torch.cat((rel_position, torch.tensor([0]))))

        sensor_position = cam_pos + rot_position[:-1]
        viewing_direction = torch.matmul(transform.transform.float(), torch.tensor([1.,0,0,0]))

        sensor = optics.Sensor(position=sensor_position, viewing_direction=tuple(viewing_direction.numpy()),
                               resolution=res, pixel_size=pixsize,
                               poisson_noise_mean=2.31, quantum_efficiency=0.72)
        
        # Add sensor and lens to the scene
        scene.add_object(sensor)
        scene.add_object(lens)
        
    return scene, n_cameras


def setup(ncams =4, deltaphi=0.2):

    #from gradoptics.optics.bounding_box import BoundingBox

    #k_fringe = 1 / (0.00001*2)
    k_fringe = 1 / (0.00003*2) # default
    #k_fringe = 1 / (0.000005*2) # no fringes from off-axis z view

    phiphase=-1.0

    cloud0 = optics.AtomCloud(position=[0., 0., 0.], k_fringe=k_fringe, phi=phiphase)
    lightsource = optics.LightSourceFromDistribution(cloud0)
    scene = optics.Scene(lightsource)


    # flat light source
    #width=0.02
    #thickness = 0.0001
    #scene = optics.Scene(optics.LightSourceFromDistribution(FlatSquareDistribution(position=[0., 0., 0.], width=width, thickness=thickness)))

    # camera position
    mode='40k'
    dth = 16.0/180.0 * np.pi # old value 21
    dph = 22.5/180.0 * np.pi # old value 15

    if ncams==5:
        camthetas= [np.pi/2, np.pi/2, 0., np.pi/16, np.pi/16]
        camphis = [0., np.pi/2., 0., np.pi/2, 3*np.pi/2]
        camrs = [5e-2, 5e-2, 5e-2, 4e-1, 4e-1]
        cammag = [0.1, 0.1, 0.1, 0.1, 0.1]
        camna = [1/1.4, 1/1.4, 1/1.4, 1/8., 1/8.]
    elif ncams==4:
        camthetas= [np.pi/2, np.pi/2, np.pi/2 + dth, np.pi/2 + dth]
        #camphis = [0. + deltaphi, np.pi/2 + deltaphi, dph + deltaphi, np.pi/2 -dph + deltaphi, ] 
        camphis = [0. + deltaphi, np.pi/2 + deltaphi, dph + deltaphi, -dph + deltaphi, ] 
        camrs = [1.75e-1, 1.75e-1, 1.75e-1, 1.75e-1,]
        cammag = [0.1, 0.1, 0.1, 0.1, ]
        camna = [1/2.0, 1/2.0, 1/2.0, 1/2.0,]
    elif ncams==3:
        #camthetas= [np.pi/2, np.pi/2,  0]
        #camphis = [0., np.pi/2., -np.pi/2]
        #camrs = [5e-2, 5e-2, 5e-2]
        #cammag = [0.1, 0.1, 0.1]
        #camna = [1/1.4, 1/1.4, 1/1.4]
        #camthetas= [np.pi/2, np.pi/16, np.pi/16]
        #camphis = [0., np.pi/2, 3*np.pi/2]
        #camrs = [5e-2, 4e-1, 4e-1]
        #cammag = [0.1, 0.1, 0.1]
        #camna = [1/1.4, 1/5.6, 1/5.6]

        camthetas= [np.pi/2, np.pi/2, np.pi/2 + dth, ]
        camphis = [0. + deltaphi, np.pi/2 + deltaphi, dph + deltaphi, ] 
        camrs = [1.75e-1, 1.75e-1, 1.75e-1, ]
        cammag = [0.1, 0.1, 0.1, ]
        camna = [1/2.0, 1/2.0, 1/2.0, ]
    elif ncams==2:
        camthetas= [np.pi/2, np.pi/2]
        camphis = [0. + deltaphi, np.pi/2. + deltaphi]
        #camrs = [5e-2, 5e-2]
        #camna = [1/2.0, 1/2.0,]
        camrs = [1.75e-1, 1.75e-1, ]
        cammag = [0.1, 0.1]
        camna = [1/1.4, 1/1.4]
    elif ncams==6:
        camthetas= [np.pi/2, np.pi/2, np.pi/2 + dth, np.pi/2 - dth, np.pi/2 + dth, np.pi/2 -dth,]
        camphis = [0. + deltaphi, np.pi/2 + deltaphi, dph + deltaphi, dph + deltaphi, -dph + deltaphi, -dph + deltaphi, ] 
        camrs = [1.75e-1, 1.75e-1, 1.75e-1, 1.75e-1, 1.75e-1, 1.75e-1]
        cammag = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,]
        camna = [1/2.0, 1/2.0, 1/2.0, 1/2.0, 1/2.0, 1/2.0]


    scene, n_cameras = setup_scene(scene, camthetas, camphis, camrs, cammag, camna, mode)

    scene.light_source.bounding_shape = optics.BoundingSphere(radii=0.002, xc=0, yc=0, zc=0)

    # bounding box for flat light source
    #scene.light_source.bounding_shape = BoundingBox(xc=0, yc=0, zc=0, dx=width, dy=width, dz=thickness)

    return scene, n_cameras

def get_pixel_coords(sensor):
    # Pixel coordinates in camera space
    x = torch.linspace(-sensor.pixel_size[0]*sensor.resolution[0]/2 + sensor.pixel_size[0]/2,
                         sensor.pixel_size[0]*sensor.resolution[0]/2 - sensor.pixel_size[0]/2, 
                         sensor.resolution[0])

    y = torch.linspace(-sensor.pixel_size[1]*sensor.resolution[1]/2 + sensor.pixel_size[1]/2,
                         sensor.pixel_size[1]*sensor.resolution[1]/2 - sensor.pixel_size[1]/2, 
                         sensor.resolution[1])
    
    pix_x, pix_y = torch.meshgrid(x, y)
    
    pix_z = torch.zeros((sensor.resolution[0], sensor.resolution[1]))
    
    all_coords = torch.stack([pix_x, pix_y, pix_z], dim=-1).reshape((-1, 3)).double()
    
    # Use transforms from above setup to go from pixel space to real (world) space
    return sensor.c2w.apply_transform_(all_coords)

def get_rays_pinhole(sensor, lens, nb_rays=None, ind=None, device='cuda', return_ind=False):
    
    if ind is None:
        #Set up for ray batching -- nb_rays switches between all pinhole rays or random batches
        if nb_rays == None or nb_rays == sensor.resolution[0]*sensor.resolution[1]:
            ind = torch.arange(0, sensor.resolution[0]*sensor.resolution[1])
        else:
            ind = torch.randint(0, sensor.resolution[0]*sensor.resolution[1], (nb_rays,))
    
    # Get origins
    all_pix_coords = get_pixel_coords(sensor)  
    origins = all_pix_coords[ind]
    
    #Get directions to center of lens
    lens_center = lens.transform.transform[:-1, -1]
    
    directions = optics.batch_vector(lens_center[None, 0] - origins[:, 0],
                                     lens_center[None, 1] - origins[:, 1],
                                     lens_center[None, 2] - origins[:, 2]).type(origins.dtype)

    # Set up rays
    rays_sensor_to_lens = Rays(origins, directions, device=device)
    
    if return_ind:
        return rays_sensor_to_lens, ind
    else:
        return rays_sensor_to_lens

def sample_on_pixels(sensor, samples_per_pixel, idx, idy, device='cpu'):
    """
    Sample data on the pixels whose id are provided

    :param samples_per_pixel: Number of samples per pixel (:obj:`int`)
    :param idx: Pixel indices along the horizontal axis, assuming a coordinate system centered in the center of
                the image(:obj:`torch.tensor`)
    :param idy: Pixel indices along the vertical axis, assuming a coordinate system centered in the center of
                the image(:obj:`torch.tensor`)
    :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

    :return: (:obj:`tuple`) Sampled points (:obj:`torch.tensor`) and p(A) (:obj:`float`)
    """

    assert idx.shape[0] == idy.shape[0]
    nb_pixels = idx.shape[0]

    samples = torch.zeros((nb_pixels, samples_per_pixel, 3), device=device)
    samples[:, :, 0] = (idx.to(device)[..., None] + 0.5) * sensor.pixel_size[0]  + (torch.rand((nb_pixels, samples_per_pixel), device=device)-0.5) * sensor.pixel_size[0]
    samples[:, :, 1] = (idy.to(device)[..., None] + 0.5) * sensor.pixel_size[1]  + (torch.rand((nb_pixels, samples_per_pixel), device=device)-0.5) * sensor.pixel_size[1]

    return (sensor.c2w.apply_transform_(samples.reshape(-1, 3)).reshape((nb_pixels, samples_per_pixel, 3)),
            1 / (sensor.resolution[0] * sensor.pixel_size[0] * sensor.resolution[1] * sensor.pixel_size[1]))

def sample_on_sensor(sensor, nb_rays, device='cpu'):
    """
    Sample data on the pixels whose id are provided

    :param nb_rays: Number of samples (:obj:`int`)

    :param device: The desired device of returned tensor (:obj:`str`). Default is ``'cpu'``

    :return: (:obj:`tuple`) Sampled points (:obj:`torch.tensor`) and p(A) (:obj:`float`)
    """
    # number of pixels
    nxpix = sensor.resolution[0]
    nypix = sensor.resolution[1]
    samples_per_pixel = 1

    randidx = torch.randint(0, nxpix, (nb_rays, ), device=device)
    randidy = torch.randint(0, nxpix, (nb_rays, ), device=device)
    #randind = randidy*sensor.resolution[0] + randidx
    randind = randidx*sensor.resolution[1] + randidy

    ones = torch.ones((nb_rays, ), dtype=torch.float64, device=device)

    samples = torch.zeros((nb_rays, 3), device=device)
    samples[:, 0] = (-nxpix/2*ones + randidx + torch.rand((nb_rays, ), device=device)-0.5 )*sensor.pixel_size[0]
    samples[:, 1] = (-nypix/2*ones + randidy + torch.rand((nb_rays, ), device=device)-0.5 )*sensor.pixel_size[1]
    

    return (sensor.c2w.apply_transform_(samples.reshape(-1, 3)).reshape((nb_rays, samples_per_pixel, 3)),
            1 / (sensor.resolution[0] * sensor.pixel_size[0] * sensor.resolution[1] * sensor.pixel_size[1]),
            randind)

def render_pixels(sensor, lens, scene, light_source, samples_per_pixel, directions_per_sample, px_i, px_j, integrator,
                  device='cpu', max_iterations=3):

    # Sampling on the sensor
    #ray_origins, p_a1 = sensor.sample_on_pixels(samples_per_pixel, px_i, px_j, device=device)
    ray_origins, p_a1 = sample_on_pixels(sensor, samples_per_pixel, px_i, px_j, device=device)
    ray_origins = ray_origins.reshape(-1, 3).double()
    ray_origins = ray_origins.expand([directions_per_sample] + list(ray_origins.shape)
                                     ).transpose(0, 1).reshape(-1, 3)

    # Sampling directions
    p_prime, p_a2 = lens.sample_points_on_lens(ray_origins.shape[0], device=device)

  

    # not normalized
    ray_directions = optics.batch_vector(p_prime[:, 0] - ray_origins[:, 0], p_prime[:, 1] - ray_origins[:, 1],
                                         p_prime[:, 2] - ray_origins[:, 2])
    ray_lenssource = optics.batch_vector(p_prime[:, 0] , p_prime[:, 1] , p_prime[:, 2])

    # @Todo, generalization needed
    sensor_normal = (lens.transform.transform[:-1, -1]-sensor.position).to(device)
    sensor_normal *= 1./torch.norm(sensor_normal)

    # distance squared of each ray
    r2 = optics.dot_product(ray_directions, ray_directions)
    #r2 = optics.dot_product(ray_lenssource, ray_lenssource)

    # orig
    cos_theta_ray = cos_theta(sensor_normal[None, ...], ray_directions)
    cos_theta_orig = cos_theta(sensor_normal[None, ...], ray_origins) 
    #cos_theta_ray = cos_theta(sensor_normal[None, ...], -ray_lenssource)

    xsperp = ray_origins - cos_theta_orig[:, None] * sensor_normal[None, :] * torch.norm(ray_origins, dim=1, keepdim=True)

    rays = Rays(ray_origins.to(device), ray_directions.to(device), meta={'cos_theta': cos_theta_ray}, device=device)

    intensities = optics.backward_ray_tracing(rays, scene, light_source, integrator, max_iterations=max_iterations)

    # Computing rendering integral

    # perpendicular distance from center of the lens

    mag = lens.m
    #sensor lens distance
    z = ((lens.transform.transform[:3, -1] - sensor.c2w.transform[:3, -1])**2).sum().sqrt()

    weight = (z/mag)**1 / torch.pow(torch.norm(p_prime + xsperp/mag, dim=1), 3)

    #intensities = (intensities * weight).reshape(px_i.shape[0], samples_per_pixel, directions_per_sample).mean(-1).mean(-1) / p_a2 /p_a1 

    # intensity proportional to area lens irrelevant to sensor size 
    intensities = (cos_theta_ray**4 * intensities ).reshape(px_i.shape[0], samples_per_pixel, directions_per_sample
                                                   ).mean(-1).mean(-1) / p_a2 /p_a1 /z**2
    
 
    return intensities



# same as above except multiple sampling per sensor
def render_pixels_samplerays(sensor, lens, scene, light_source, nb_rays, directions_per_sample, integrator,
                  device='cpu', max_iterations=3):

    # Sampling on the sensor
    ray_origins, p_a1, ind = sample_on_sensor(sensor, nb_rays,  device=device)
    ray_origins = ray_origins.reshape(-1, 3).double()
    ray_origins = ray_origins.expand([directions_per_sample] + list(ray_origins.shape)
                                     ).transpose(0, 1).reshape(-1, 3)

    # Sampling directions
    p_prime, p_a2 = lens.sample_points_on_lens(ray_origins.shape[0], device=device)

  

    # not normalized
    ray_directions = optics.batch_vector(p_prime[:, 0] - ray_origins[:, 0], p_prime[:, 1] - ray_origins[:, 1],
                                         p_prime[:, 2] - ray_origins[:, 2])

    # @Todo, generalization needed
    sensor_normal = (lens.transform.transform[:-1, -1]-sensor.position).to(device)
    sensor_normal *= 1./torch.norm(sensor_normal)

    # distance squared of each ray


    # orig
    cos_theta_ray = cos_theta(sensor_normal[None, ...], ray_directions)
    cos_theta_orig = cos_theta(sensor_normal[None, ...], ray_origins) 

    xsperp = ray_origins - cos_theta_orig[:, None] * sensor_normal[None, :] * torch.norm(ray_origins, dim=1, keepdim=True)

    rays = Rays(ray_origins.to(device), ray_directions.to(device), meta={'cos_theta': cos_theta_ray}, device=device)

    intensities = optics.backward_ray_tracing(rays, scene, light_source, integrator, max_iterations=max_iterations)

    # Computing rendering integral

    # perpendicular distance from center of the lens

    mag = lens.m
    #sensor lens distance
    z = ((lens.transform.transform[:3, -1] - sensor.c2w.transform[:3, -1])**2).sum().sqrt()

    #weight = (z/mag)**1 / torch.pow(torch.norm(p_prime + xsperp/mag, dim=1), 3)

    #intensities = (intensities * weight).reshape(nb_rays, directions_per_sample).mean(-1) / p_a2 /p_a1 
    intensities = (cos_theta_ray**4 * intensities ).reshape(nb_rays, directions_per_sample
                                                   ).mean(-1) / p_a2 /p_a1 /z**2
 
 
    return intensities, ind


def calc_backward(scene, n_cameras):

    # 32 uniformly spaced points, 32 additional points
    integrator = HierarchicalSamplingIntegrator(32, 32)

    # Loop over cameras
    targets_backward = []
    use_pinhole = False

    nloops = 10

    if use_pinhole:
        # pinhole camera
        for i_cam in tqdm(range(n_cameras)):
            # Generate rays for each camera (all pixels at once). i_cam*2 gives all sensor idxs, lenses are i_cam*2+1
            incident_rays = get_rays_pinhole(scene.objects[i_cam*2], scene.objects[i_cam*2+1])
            
            # Trace rays through the scene (includes the integration)
            intensities = optics.backward_ray_tracing(incident_rays, scene, 
                                                        scene.light_source, integrator, max_iterations=3)
            
            # Store the result for a given camera
            targets_backward.append(intensities.cpu().clone().reshape(scene.objects[i_cam*2].resolution))
            
            del intensities
    else:
        for i_cam in tqdm(range(n_cameras)):
            sensor = scene.objects[i_cam*2]
            lens = scene.objects[i_cam*2+1]
            indicesx, indicesy = torch.arange(-sensor.resolution[0]/2, sensor.resolution[0]/2), torch.arange(-sensor.resolution[1]/2, sensor.resolution[1]/2)
            meshidx, meshidy = torch.meshgrid(indicesx, indicesy)
            meshidx = meshidx.reshape((sensor.resolution[0]*sensor.resolution[1]))
            meshidy = meshidy.reshape((sensor.resolution[0]*sensor.resolution[1]))
            
            sumintensities = torch.zeros((sensor.resolution[0], sensor.resolution[1]), dtype=torch.float64, device='cuda')
            for _ in range(nloops):
                # render pixels for target
                intensity = render_pixels(sensor, lens, scene, scene.light_source, 1, 1, meshidx, meshidy, integrator, device='cuda')
                #sumintensities += intensity.cpu().clone().reshape((sensor.resolution[0], sensor.resolution[1])) / nloops
                sumintensities += intensity.reshape((sensor.resolution[0], sensor.resolution[1])) / nloops

                # random batch sampling
                #intensity, ind = render_pixels_samplerays(sensor, lens, scene, scene.light_source, 2048, 1, integrator, device='cuda')
                #sumintflat = sumintensities.flatten()
                #sumintflat[ind.to('cpu')] += intensity.cpu().clone() / nloops
                #sumintensities = sumintflat.reshape((sensor.resolution[0], sensor.resolution[1]))
                #sumintensities += intensity.cpu().numpy().reshape((sensor.resolution[0], sensor.resolution[1])) / nloops
                
                del intensity        
            targets_backward.append(sumintensities.cpu().clone())
            del sumintensities

    #intensities_backward = [np.sum(targets_backward[i].cpu().numpy()) for i in range(len(targets_backward))]
    intensities_backward = [np.sum(targets_backward[i].numpy()) for i in range(len(targets_backward))]

    return targets_backward, intensities_backward



def calc_forward(scene, ):
    # adaptive sampling
    from gradoptics.ray_tracing import ray_tracing

    ntotalrays=1e7
    nrayspertrial=1e6
    ntrials = int(ntotalrays/nrayspertrial)

    directionslist = []
    rawdatalist = []
    for itrials in tqdm(range(ntrials)):
        raystrial = scene.light_source.sample_rays(int(nrayspertrial), device='cuda')
        _, _, mask = ray_tracing.trace_rays(raystrial, scene)
        directionslist.append(raystrial[mask].directions.cpu().numpy().astype(np.float32))
        rawdatalist.append(raystrial.directions.cpu().numpy().astype(np.float32))
    directionsnp = np.vstack(directionslist).astype(np.float32)
    rawdatanp = np.vstack(rawdatalist).astype(np.float32)
    #rawdatanp = raystrial.directions.numpy()
   
    centroids, maxdist, counts, counts_all = kwindows(directionsnp, rawdatanp, directionsnp)

    def importance_sampledrays(nb_rays, device='cpu'):
        directionvectors,_ = weighted_sampler(nb_rays, centroids, maxdist, counts_all)
        origins = scene.light_source.distribution.sample(nb_rays, device=device)
        
        directions = normalize_batch_vector(torch.Tensor(directionvectors))
        torch.cuda.empty_cache()
        return Rays(origins, directions, device=device)
    
    @torch.no_grad()
    def make_image(scene, device='cpu', nb_rays=int(2e6), batch_size=int(1e5), quantum_efficiency=True, max_iterations=2,
                add_poisson_noise=True, lookup_table=None, show_progress=True, destructive_readout=True):
        
        progress_bar = tqdm if show_progress else lambda x: x
        nb_rays_left_to_sample = nb_rays
        for _ in progress_bar(range(int(np.ceil(nb_rays / batch_size)))):
            #rays = scene.light_source.sample_rays(min(batch_size, nb_rays_left_to_sample), device=device)
            rays = importance_sampledrays(min(batch_size, nb_rays_left_to_sample), device=device)
            optics.forward_ray_tracing(rays, scene, max_iterations=max_iterations)
            nb_rays_left_to_sample -= batch_size

            del rays
            torch.cuda.empty_cache()

        #return scene.objects[0].readout(add_poisson_noise=add_poisson_noise, destructive_readout=destructive_readout)
        pass
    nphotonsfromcloud = 1e9
    nbrays_forward=1e8 # restricted


    nphotons = np.sum(counts_all)*(nphotonsfromcloud/ntotalrays) # nphotons in acceptance per 1e9 generated
    scalefactor = nbrays_forward/nphotons
    print(scalefactor)

    #make_image(scene, nb_rays=int(1e10), batch_size=int(1e7), device='cuda')
    make_image(scene, nb_rays=int(nbrays_forward), batch_size=int(5e6), device='cuda') # reweighted
    
    usenoise = False
    allsensors = get_sensors(scene)
    targets_forward = [allsensors[i].readout(add_poisson_noise = usenoise, destructive_readout = True).cpu().clone().reshape(allsensors[i].resolution).numpy() for i in range(len(allsensors))]
    # need to flip ...
    targets_forward = [np.flip(tgt) for tgt in targets_forward]
    intensities_forward = [np.sum(targets_forward[i]) for i in range(len(targets_forward))]

    return targets_forward, intensities_forward, scalefactor

def poisson_sample(targets_forward, intensities_forward, targets_backward, intensities_backward, scalefactor, ncams):
    useforwardsample = False

    #scalefactor = 1.0e9/(330000.0*10.0)
    targets_sample_np = targets_real = [None]*ncams
    noisemean = 2.0
    for i in range(len(targets_forward)):
        if useforwardsample:
            targets_real[i] = targets_forward[i].T / scalefactor + noisemean
        else:
            targets_real[i] = targets_backward[i] * intensities_forward[i] / intensities_backward[i] / scalefactor + noisemean
        targets_sample_np[i] = np.random.poisson(targets_real[i], None).astype(np.float64)
        

    targets_sample = []
    for a in targets_sample_np:
        targets_sample.append(torch.tensor(a))
    
    return targets_sample

def train(model, scene, ncams, targets, intensities, nloops):
    losses = []
    sample_weights = np.array(intensities) 
    sample_weights /= np.sum(sample_weights)
    # copy targets to cuda
    targets_forward_cuda = []
    usepinhole = False
    batch_size = 1024

    # Loss function -- mean squared error between pixels
    #loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.PoissonNLLLoss(log_input=False)

    # Same integrator as above
    integrator = HierarchicalSamplingIntegrator(32, 16)
    lrsc = 5
    noisemean=2.0
    paramslr = [
        {'params':[model.pos], 'name':'pos', 'lr':1e-5*lrsc},
        {'params':[model.covmat], 'name':'covmat', 'lr':1e-4*lrsc},
        {'params':[model.u], 'name':'u', 'lr':1e-5*lrsc},
        {'params':[model.v], 'name':'v', 'lr':1e-5*lrsc},
        {'params':[model.w], 'name':'w', 'lr':1e-5*lrsc},
        {'params':[model.contrast], 'name':'contrast', 'lr':1e-2*lrsc},
        {'params':[model.k], 'name':'k', 'lr':1e-2*lrsc},
        {'params':[model.phi], 'name':'phi', 'lr':1e-3*lrsc},
        {'params':[model.intensity], 'name':'intensity', 'lr':1e-2*lrsc},
    ]
    optimizer = torch.optim.Adam(paramslr, lr=0.0)
    for i_cam in range(ncams):
        targets_forward_cuda.append(targets[i_cam].flatten().to('cuda'))

    for i_iter in range(nloops):
        # Take a few samples from each camera put them in the list
        intensities_list = []
        target_list = []
        for i_cam in range(ncams):
            sensor = scene.objects[i_cam*2]
            lens = scene.objects[i_cam*2+1]
            if usepinhole:
                # Grab a random batch of rays
                rays, ind = get_rays_pinhole(sensor, lens, nb_rays=batch_size, device='cuda', return_ind=True)
                # Ray trace using neural network light source
                intensities_cam = optics.backward_ray_tracing(rays, scene, 
                                                        scene.light_source, integrator, max_iterations=3)
            else:
                intensities_cam, ind = render_pixels_samplerays(sensor, lens, scene, scene.light_source, batch_size, 1, integrator, device='cuda')
            # Get corresponding pixels from target images
            #target_vals_cam = targets_forward_sample[i_cam].flatten().to('cuda')[ind]
            target_vals_cam = targets_forward_cuda[i_cam][ind]
            

            intensities_list.append(intensities_cam)
            target_list.append(target_vals_cam)
            
        
        intensities = torch.vstack(intensities_list)
        target_vals = torch.vstack(target_list)
        
        # Calculate the loss -- 1e9 scaling is a result of unnormalized PDF in atom cloud
        loss = 0.0
        for i_cam in range(ncams):
            loss += sample_weights[i_cam] * loss_fn(intensities_list[i_cam] + noisemean, target_list[i_cam])
            #loss += sample_weights[i_cam] * loss_fn(1e6*intensities_list[i_cam] + noisemean, target_list[i_cam]) # why this works better?
            #loss += loss_fn(1e6*intensities_list[i_cam] + noisemean, target_list[i_cam]) # try
        # Calculate gradients and update network parameters
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Keep track of results
        losses.append(loss.item())
        del intensities
        del target_vals
    pass

# Main driver for scans
def drive(ncams, azim, nsteps, nexperiments, dirname='multicloudvar', initpars=[[[0., 0., 0.]], [0.2, 0.2, 0.2], 25.0, 0.5, 1/0.03, 0.0]):
    Path(dirname).mkdir(exist_ok=True) # create directory if it doesn't exist 

    fname = f'{dirname}/multi_{ncams}_{azim}_{nsteps}.pkl'
    cols = ['ncams', 'azim', 'alpha', 'k', 'phi', 'covmat']
    if pathlib.Path(fname).exists():
        dfmulti = pd.read_pickle('multi.pkl')
    else:
        dfmulti = pd.DataFrame(columns=cols)

    scene, _ = setup(ncams, azim)
    targets_backward, intensities_backward = calc_backward(scene, ncams)
    print(intensities_backward)
    targets_forward, intensities_forward, scalefactor = calc_forward(scene)
    print(intensities_forward)
    phis = []
    alphas = []
    ks=[]
    covmats = []

    device = 'cuda'
    # Batch size

    params = initpars + [device]
    for _ in tqdm(range(nexperiments)):
        # poisson sampled target
        targets = poisson_sample(targets_forward, intensities_forward, targets_backward, intensities_backward, scalefactor, ncams)
        # NN light source using slightly tighter bounding sphere
        #model = parametrizedcloud([[0., 0., 0.]], [0.2, 0.2, 0.2], 25.0, 0.5, 1/0.03, 0.0, device)
        model = parametrizedcloud(*params)
        nn_light_source = LightSourceFromNeuralNet(model, optics.BoundingSphere(radii=0.003, 
                                                                                xc=0, yc=0, zc=0),
                                                            rad=0.003, x_pos=0)
        scene_train = optics.Scene(nn_light_source)

        # Add sensors/lenses from scene above
        for anobj in scene.objects:
            scene_train.add_object(anobj)
        train(model, scene_train, ncams, targets, intensities_forward, nsteps)
        phi = model.phi.clone().cpu().detach().numpy()
        phis.append(phi)
        alpha = torch.nn.Sigmoid()(model.contrast.clone().cpu().detach()).numpy()
        alphas.append(alpha)
        k = model.k.clone().cpu().detach().numpy()
        ks.append(k)
        cmat = model.covmat.clone().cpu().detach().numpy()
        #dfmulti.append([ncams, azim, alpha, k, phi])
        dfmulti = pd.concat([dfmulti, pd.DataFrame([[ncams, azim, alpha, k, phi, cmat]], columns=cols)], ignore_index=True)
    print(f'phi: {np.average(phis)} +- {np.std(phis, ddof=1)}')
    print(f'alpha: {np.average(alphas)} +- {np.std(alphas, ddof=1)}')
    print(f'k: {np.average(ks)} +- {np.std(ks, ddof=1)}')
    dfmulti.to_pickle(fname)
    pass

def run():
    scene, ncams = setup(2)
    targets_backward, intensities_backward = calc_backward(scene, ncams)
    print(intensities_backward)
    targets_forward, intensities_forward, scalefactor = calc_forward(scene)
    print(intensities_forward)

    nexperiments = 10
    device = 'cuda'
    # Batch size
    phis = []
    alphas = []
    for _ in tqdm(range(nexperiments)):
        # poisson sampled target
        targets = poisson_sample(targets_forward, intensities_forward, targets_backward, intensities_backward, scalefactor, ncams)
        model = parametrizedcloud([[0., 0., 0.]], [0.6, 0.6, 0.6], 25.0, 0.5, 1/0.03, 0.0, device)
        # NN light source using slightly tighter bounding sphere
        nn_light_source = LightSourceFromNeuralNet(model, optics.BoundingSphere(radii=0.003, 
                                                                                xc=0, yc=0, zc=0),
                                                            rad=0.003, x_pos=0)
        scene_train = optics.Scene(nn_light_source)

        # Add sensors/lenses from scene above
        for anobj in scene.objects:
            scene_train.add_object(anobj)
        train(model, scene_train, ncams, targets, intensities_forward, 3000)
        phis.append(model.phi.clone().cpu().detach().numpy())
        alphas.append(model.contrast.clone().cpu().detach().numpy())
    print(f'phi: {np.average(phis)} +- {np.std(phis, ddof=1)}')
    print(f'alpha: {np.average(alphas)} +- {np.std(alphas, ddof=1)}')
    pass

if __name__=='__main__':
    run()
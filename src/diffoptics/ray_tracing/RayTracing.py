from typing import Tuple

import torch
import warnings

import diffoptics as optics
from diffoptics.optics import Rays
from diffoptics.ray_tracing.Scene import Scene


def trace_rays(incident_rays, scene):
    """
    Performs one iteration of ray tracing, i.e. finds the closest object in the ``scene`` each ray will intersect, and
    computes the reflected or refracted rays. Rays with ``nan`` components are returned for rays that do not have
    intersections

    :param incident_rays: Batch of incident rays (:py:class:`~diffoptics.optics.Ray.Rays`)
    :param scene: Scene in which the rays are travelling (:py:class:`~diffoptics.ray_tracing.Scene.Scene`)

    :return: (:obj:`tuple`)

             - Reflected or refracted rays after one iteration of ray tracing (:py:class:`~diffoptics.optics.Ray.Rays`)
             - Times at which the rays intersect the closest object to them and ``nan`` for rays that have no
               intersection (:obj:`torch.tensor`)
             - Boolean tensor that indicates of the intersected objects are lenses (:obj:`torch.tensor`)
    """
    device = incident_rays.device
    t = torch.zeros(incident_rays.origins.shape[0], device=device) + float('Inf')
    outgoing_ray = optics.empty_like(incident_rays)
    is_lens = torch.empty(incident_rays.origins.shape[0], device=device) + float('nan')

    for o in scene.objects:
        t_current = o[0].get_ray_intersection(incident_rays)
        # If there is an intersection and if the intersected object is the first one the ray intersects, update the
        # return values
        condition = (t_current < t) & (t_current > 1e-3) & (
            ~torch.isnan(t_current))  # & (not jnp.isnan(t_for_current_object))
        t[condition] = t_current[condition].type(t.dtype)
        rays = o[0].intersect(incident_rays[condition], t_current[condition])
        outgoing_ray.origins[condition] = rays.origins
        outgoing_ray.directions[condition] = rays.directions
        if outgoing_ray.luminosities is not None:
            outgoing_ray.luminosities[condition] = rays.luminosities
        is_lens[condition] = o[1]

    t[torch.isinf(t)] = float('nan')
    return outgoing_ray, t, is_lens


def forward_ray_tracing(incident_rays, scene, max_iterations=2, ax=None, quantum_efficiency=True):
    """
    Performs forward ray tracing, i.e. computes the path taken by the rays ``incident_rays`` in the scene ``scene``
    until the maximum number of bounces ``max_iterations`` is reached, or until the rays hit a sensor

    :param incident_rays: Rays that should be traced in the scene (:py:class:`~diffoptics.optics.Ray.Rays`)
    :param scene: Scene in which the rays are travelling (:py:class:`~diffoptics.ray_tracing.Scene.Scene`)
    :param max_iterations: Maximum number of bounces in the ``scene`` (:obj:`int`)
    :param ax: 3d axes in which the rays are plotted (if ``ax`` is not ``None``) as they traverse the scene
               (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`). Default is ``None``
    :param quantum_efficiency: A boolean that indicates if quantum efficiency should be used when the rays hit a sensor.
                               Default is ``True``

    :return: (:obj:`tuple`)

             - Positions where the rays hit a sensor (in world-space) (:obj:`torch.tensor`)
             - Luminosities carried by the rays (:obj:`torch.tensor`)
             - Meta data stored in the rays (:obj:`dict`)
    """

    rays_lens_camera = None

    for i in range(max_iterations):
        outgoing_rays, t, are_lens = trace_rays(incident_rays, scene)

        # Only keep the rays that intersect with the system
        index = ~torch.isnan(t)
        outgoing_rays = outgoing_rays[index]
        are_lens = are_lens[index].type(torch.bool)

        # if ax is not None:  # Plot ray
        #    for j in range(outgoing_ray.shape[0]):
        #        ray.plot_ray(ax, incident_rays[j], t[j], line_width=0.2)

        tmp_rays_lens_camera = outgoing_rays[are_lens]  # rays refracted by the lens, to track up to the sensor
        rays_lens_camera = optics.cat(rays_lens_camera, tmp_rays_lens_camera) if rays_lens_camera is not None else \
            tmp_rays_lens_camera
        incident_rays = outgoing_rays[~are_lens]  # rays that still need to be traced

        del tmp_rays_lens_camera
        torch.cuda.empty_cache()

    # Time at which the rays will hit the sensor
    t_camera = scene.sensor.get_ray_intersection(rays_lens_camera)
    index = ~torch.isnan(t_camera)
    rays_lens_camera = rays_lens_camera[index]
    t_camera = t_camera[index]

    # if rays_lens_camera.origins.shape[0] == 0:
    #    continue

    # if ax is not None:  # Plot ray
    #    for j in range(rays_lens_camera.shape[0]):
    #        ray.plot_ray(ax, rays_lens_camera[j], t_camera[j], line_width=0.2)

    # energy deposit at sensor
    hit_position, luminosities, meta = scene.sensor.intersect(rays_lens_camera, t_camera,
                                                              quantum_efficiency=quantum_efficiency)

    if incident_rays.origins.shape[0] > 0:
        warnings.warn("The maximum number of iteration has been reached and there are still rays"
                      "bouncing in the scene. incident_rays.shape[0]: " + str(incident_rays.origins.shape[0]))

    del t_camera
    del index
    del rays_lens_camera
    torch.cuda.empty_cache()

    return hit_position, luminosities, meta


def backward_ray_tracing(incident_rays, scene):
    raise NotImplemented

from typing import Tuple

import torch
import warnings

from diffoptics.optics import Rays
from diffoptics.ray_tracing.Scene import Scene


def trace_rays(incident_rays: Rays, scene: Scene) -> Tuple[Rays, torch.Tensor, torch.Tensor]:
    """
    @Todo
    :param incident_rays:
    :param scene:
    :return:
    """
    device = incident_rays.device
    t = torch.empty(incident_rays.origins.shape[0], device=device) + float('Inf')
    outgoing_ray = Rays(torch.empty(incident_rays.origins.shape[0], 3) + float('nan'),
                        torch.empty(incident_rays.origins.shape[0], 3) + float('nan'),
                        luminosities=(torch.empty(incident_rays.origins.shape[0],
                                                  dtype=incident_rays.luminosities.dtype) + float('nan')) if
                        incident_rays.luminosities is not None else None, device=device)
    is_lens = torch.empty(incident_rays.origins.shape[0], device=device) + float('nan')

    for o in scene.objects:
        t_current = o[0].get_ray_intersection(incident_rays)
        # If there is an intersection and if the intersected object is the first one the ray intersects, update the
        # return values
        condition = (t_current < t) & (t_current > 1e-3) & (
            ~torch.isnan(t_current))  # & (not jnp.isnan(t_for_current_object))
        t[condition] = t_current[condition].type(t.dtype)
        rays = o[0].intersect(incident_rays.get_at(condition), t_current[condition])
        outgoing_ray.origins[condition] = rays.origins
        outgoing_ray.directions[condition] = rays.directions
        if outgoing_ray.luminosities is not None:
            outgoing_ray.luminosities[condition] = rays.luminosities
        is_lens[condition] = o[1]

    t[torch.isinf(t)] = float('nan')
    return outgoing_ray, t, is_lens


def forward_ray_tracing(incident_rays: Rays, scene: Scene, max_iterations=2, ax=None, quantum_efficiency=True) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    rays_lens_camera_origins = torch.tensor([], device=incident_rays.device)
    rays_lens_camera_directions = torch.tensor([], device=incident_rays.device)
    rays_lens_camera_luminosities = torch.tensor([], device=incident_rays.device) if \
        incident_rays.luminosities is not None else None

    for i in range(max_iterations):
        outgoing_rays, t, are_lens = trace_rays(incident_rays, scene)

        # Only keep the rays that intersect with the system
        index = ~torch.isnan(t)
        outgoing_rays = outgoing_rays.get_at(index)
        are_lens = are_lens[index].type(torch.bool)

        # if ax is not None:  # Plot ray
        #    for j in range(outgoing_ray.shape[0]):
        #        ray.plot_ray(ax, incident_rays[j], t[j], line_width=0.2)

        tmp_rays_lens_camera = outgoing_rays.get_at(are_lens)  # rays refracted by the lens, to track up to the sensor
        rays_lens_camera_origins = torch.cat((rays_lens_camera_origins, tmp_rays_lens_camera.origins))
        rays_lens_camera_directions = torch.cat((rays_lens_camera_directions, tmp_rays_lens_camera.directions))
        if rays_lens_camera_luminosities is not None:
            rays_lens_camera_luminosities = torch.cat(
                (rays_lens_camera_luminosities, tmp_rays_lens_camera.luminosities))
        incident_rays = outgoing_rays.get_at(~are_lens)  # rays that still need to be traced

        del tmp_rays_lens_camera
        torch.cuda.empty_cache()

    # Time at which the rays will hit the sensor
    rays_lens_camera = Rays(rays_lens_camera_origins,
                            rays_lens_camera_directions,
                            luminosities=rays_lens_camera_luminosities,
                            device=incident_rays.device)
    t_camera = scene.sensor.get_ray_intersection(rays_lens_camera)
    index = ~torch.isnan(t_camera)
    rays_lens_camera = rays_lens_camera.get_at(index)
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

    del rays_lens_camera_origins
    del rays_lens_camera_directions
    del rays_lens_camera_luminosities
    del t_camera
    del index
    del rays_lens_camera
    torch.cuda.empty_cache()

    return hit_position, luminosities, meta


def backward_ray_tracing(incident_rays: Rays, scene: Scene):
    raise NotImplemented

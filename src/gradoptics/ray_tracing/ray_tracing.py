import torch
import warnings
import gradoptics as optics


def trace_rays(incident_rays, scene):
    """
    Performs one iteration of ray tracing, i.e. finds the closest object in the ``scene`` each ray will intersect, and
    computes the reflected or refracted rays. Rays with ``nan`` components are returned for rays that do not have
    intersections

    :param incident_rays: Batch of incident rays (:py:class:`~gradoptics.optics.ray.Rays`)
    :param scene: Scene in which the rays are travelling (:py:class:`~gradoptics.ray_tracing.scene.Scene`)

    :return: (:obj:`tuple`)
             - Reflected or refracted rays after one iteration of ray tracing (:py:class:`~gradoptics.optics.ray.Rays`)
             - Times at which the rays intersect the closest object to them and ``nan`` for rays that have no
               intersection (:obj:`torch.tensor`)
             - Boolean tensor that indicates if the ``incident_rays`` have been reflected/refracted or no
               (:obj:`torch.tensor`)
    """
    device = incident_rays.device
    t = torch.zeros(incident_rays.origins.shape[0], device=device) + float('Inf')
    outgoing_rays = optics.empty_like(incident_rays)
    obj_indices = torch.zeros(incident_rays.origins.shape[0], device=device, dtype=torch.long) - 1
    mask = torch.zeros(incident_rays.origins.shape[0], device=device, dtype=torch.bool)

    # Find first object intersected by each ray
    for idx, o in enumerate(scene.objects):
        t_current = o.get_ray_intersection(incident_rays)
        # If there is an intersection and if the intersected object is the first one the ray intersects, update the
        # return values
        condition = (t_current < t) & (t_current > 1e-3) & (~torch.isnan(t_current))
        t[condition] = t_current[condition].type(t.dtype)
        obj_indices[condition] = idx
        mask[condition] = 1

    # Compute the rays reflected / refracted by their intersections with the objects in the scene
    for idx, o in enumerate(scene.objects):
        condition = (obj_indices == idx)

        if condition.sum() == 0:
            continue

        tmp, mask[condition] = o.intersect(incident_rays[condition], t[condition])

        if tmp is not None:
            outgoing_rays[condition & mask] = tmp

    return outgoing_rays, t, mask


def forward_ray_tracing(incident_rays, scene, max_iterations=2, ax=None):
    """
    Performs forward ray tracing, i.e. computes the path taken by the rays ``incident_rays`` in the scene ``scene``
    until the maximum number of bounces ``max_iterations`` is reached, or until the rays hit a sensor

    :param incident_rays: Rays that should be traced in the scene (:py:class:`~gradoptics.optics.ray.Rays`)
    :param scene: Scene in which the rays are travelling (:py:class:`~gradoptics.ray_tracing.scene.Scene`)
    :param max_iterations: Maximum number of bounces in the ``scene`` (:obj:`int`)
    :param ax: 3d axes in which the rays are plotted (if ``ax`` is not ``None``) as they traverse the scene
               (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`). Default is ``None``
    """

    for i in range(max_iterations):
        outgoing_rays, t, mask = trace_rays(incident_rays, scene)

        if ax is not None:
            origins = incident_rays.origins[mask].data.cpu().numpy()
            destinations = incident_rays[mask](t[mask]).data.cpu().numpy()
            for ray_idx in range(origins.shape[0]):
                ax.plot([origins[ray_idx, 0], destinations[ray_idx, 0]],
                        [origins[ray_idx, 1], destinations[ray_idx, 1]],
                        [origins[ray_idx, 2], destinations[ray_idx, 2]])

        # Only keep the rays that intersect with the system
        incident_rays = outgoing_rays[mask]

        torch.cuda.empty_cache()

    if incident_rays.origins.shape[0] > 0:
        warnings.warn("The maximum number of iteration has been reached and there are still rays"
                      "bouncing in the scene. incident_rays.shape[0]: " + str(incident_rays.origins.shape[0]))


def backward_ray_tracing(incident_rays, scene, light_source, integrator, max_iterations=2):
    """
    Performs backward ray tracing, i.e. computes the path taken by the rays ``incident_rays`` in the scene ``scene``
    until the maximum number of bounces ``max_iterations`` is reached, or until the rays hit the light source
    ``light_source``

    :param incident_rays: Rays that should be traced in the scene (:py:class:`~gradoptics.optics.ray.Rays`)
    :param scene: Scene in which the rays are travelling (:py:class:`~gradoptics.ray_tracing.scene.Scene`)
    :param light_source: Light source (:py:class:`~gradoptics.light_sources.base_light_source.BaseLightSource`)
    :param integrator: An integrator to compute line integrals
                       (:py:class:`~gradoptics.integrator.base_integrator.BaseIntegrator`)
    :param max_iterations: Maximum number of bounces in the ``scene`` (:obj:`int`)

    :return: the intensity carried by the rays ``incident_rays`` (:py:class:`~gradoptics.optics.ray.Rays`)
    """

    intensities = torch.zeros(incident_rays.get_size(), dtype=torch.double)

    # Labelling the rays
    incident_rays.meta['track_idx'] = torch.linspace(0, incident_rays.get_size() - 1, incident_rays.get_size(),
                                                     dtype=torch.long)

    for i in range(max_iterations):
        outgoing_rays, t, mask = trace_rays(incident_rays, scene)

        # Potential intersection with the light source
        t_min, t_max = light_source.bounding_shape.get_ray_intersection_(incident_rays)

        # Mask for the rays that hit the light source rather than the object found in trace_rays
        new_mask = (t_min < t) & (t_min < t_max)

        # Computing the intensities for the rays that have hit the light source
        if new_mask.sum() > 0:
            intensities[incident_rays.meta['track_idx'][new_mask]] = integrator.compute_integral(
                incident_rays[new_mask], light_source.pdf, t_min[new_mask], t_max[new_mask]
            ) * incident_rays[new_mask].luminosities

        # Rays that are still in the scene, and have not hit the light source
        incident_rays = outgoing_rays[mask & (~new_mask)]

    return intensities


def render_pixels(sensor, lens, scene, light_source, samples_per_pixel, directions_per_sample, px_i, px_j, integrator,
                  device='cpu', max_iterations=3):

    # Sampling on the sensor
    ray_origins, p_a1 = sensor.sample_on_pixels(samples_per_pixel, px_i, px_j, device=device)
    ray_origins = ray_origins.reshape(-1, 3)
    ray_origins = ray_origins.expand([directions_per_sample] + list(ray_origins.shape)
                                     ).transpose(0, 1).reshape(-1, 3)

    # Sampling directions
    p_prime, p_a2 = lens.sample_points_on_lens(ray_origins.shape[0])
    ray_directions = optics.batch_vector(p_prime[:, 0] - ray_origins[:, 0], p_prime[:, 1] - ray_origins[:, 1],
                                         p_prime[:, 2] - ray_origins[:, 2])

    # @Todo, generalization needed
    sensor_normal = torch.tensor([1, 0, 0], device=device)
    cos_theta = optics.optics.vector.cos_theta(sensor_normal[None, ...], ray_directions)

    rays = optics.Rays(ray_origins.type(torch.float).to(device), ray_directions.type(torch.float).to(device),
                       meta={'cos_theta': cos_theta}, device=device)

    intensities = optics.backward_ray_tracing(rays, scene, light_source, integrator, max_iterations=max_iterations)

    # Computing rendering integral
    z = lens.transform.transform[0, -1] - sensor.c2w.transform[0, -1]
    intensities = (cos_theta**4 * intensities).reshape(px_i.shape[0], samples_per_pixel, directions_per_sample
                                                       ).mean(-1).mean(-1) / z**2 / p_a1 / p_a2
    return intensities


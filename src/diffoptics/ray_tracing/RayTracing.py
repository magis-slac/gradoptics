import torch
import warnings
import diffoptics as optics


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

        tmp = o.intersect(incident_rays[condition], t[condition])

        if tmp is None:  # Some optical element may not reflect or refract rays (e.g. sensor)
            mask[condition] = 0  # No rays returned
        else:
            outgoing_rays[condition] = tmp

    return outgoing_rays, t, mask


def forward_ray_tracing(incident_rays, scene, max_iterations=2, ax=None):
    """
    Performs forward ray tracing, i.e. computes the path taken by the rays ``incident_rays`` in the scene ``scene``
    until the maximum number of bounces ``max_iterations`` is reached, or until the rays hit a sensor

    :param incident_rays: Rays that should be traced in the scene (:py:class:`~diffoptics.optics.Ray.Rays`)
    :param scene: Scene in which the rays are travelling (:py:class:`~diffoptics.ray_tracing.Scene.Scene`)
    :param max_iterations: Maximum number of bounces in the ``scene`` (:obj:`int`)
    :param ax: 3d axes in which the rays are plotted (if ``ax`` is not ``None``) as they traverse the scene
               (:py:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`). Default is ``None``
    """

    for i in range(max_iterations):
        outgoing_rays, t, mask = trace_rays(incident_rays, scene)

        # Only keep the rays that intersect with the system
        incident_rays = outgoing_rays[mask]

        torch.cuda.empty_cache()

    if incident_rays.origins.shape[0] > 0:
        warnings.warn("The maximum number of iteration has been reached and there are still rays"
                      "bouncing in the scene. incident_rays.shape[0]: " + str(incident_rays.origins.shape[0]))


def backward_ray_tracing(incident_rays, scene):
    raise NotImplemented

import torch
import matplotlib.pyplot as plt
import diffoptics as optics
from diffoptics.optics import batch_vector
from diffoptics.optics.Vector import cross_product
from diffoptics.transforms.Transforms import get_look_at_transform


def _test_ray(dim=1000):
    o = batch_vector(torch.zeros(dim), torch.zeros(dim), torch.zeros(dim))
    d = batch_vector(torch.ones(dim), torch.zeros(dim), torch.zeros(dim))
    ray = optics.Rays(o, d)

    if ((ray.origins.shape[0] == ray.directions.shape[0] == dim) and (
            ray.origins.shape[1] == ray.directions.shape[1] == 3)):
        return 0
    else:
        return -1


def _test_rejection_sampling():
    # Atom cloud
    atom_cloud = optics.AtomCloud()

    # Define a sampler to sample from the cloud density
    proposal_dist = optics.GaussianDistribution(mean=0., std=0.0002)
    x = optics.rejection_sampling(atom_cloud.marginal_cloud_density_x, int(1e6), proposal_dist, m=None)
    y = optics.rejection_sampling(atom_cloud.marginal_cloud_density_y, int(1e6), proposal_dist, m=None)
    z = optics.rejection_sampling(atom_cloud.marginal_cloud_density_z, int(1e6), proposal_dist, m=None)

    plt.hist(x.numpy(), bins=300, histtype='step', color='k', label='x')
    plt.hist(y.numpy(), bins=300, histtype='step', color='C0', label='y')
    plt.hist(z.numpy(), bins=300, histtype='step', color='b', label='z')
    plt.legend()
    plt.savefig('test_rejection_sampling.pdf')
    plt.close()

    return 0


def _test_atom_cloud(nb_atoms=int(1e4)):
    # Atom cloud
    atom_cloud = optics.LightSourceFromDistribution(optics.AtomCloud())

    rays = atom_cloud.sample_rays(nb_atoms)

    origins = rays.origins
    directions = rays.directions

    if not ((origins.shape[0] == directions.shape[0] == nb_atoms) and (origins.shape[1] == directions.shape[1] == 3)):
        return -1

    # Plot the origin of the rays (the cloud)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], alpha=.3, s=0.5)
    ax.view_init(azim=0, elev=0)
    ax.set_zlim([origins[:, 2].mean() - 3 * origins[:, 2].std(), origins[:, 2].mean() + 3 * origins[:, 2].std()])
    ax.set_ylim([origins[:, 1].mean() - 3 * origins[:, 1].std(), origins[:, 1].mean() + 3 * origins[:, 1].std()])
    ax.set_xlim([origins[:, 0].mean() - 3 * origins[:, 0].std(), origins[:, 0].mean() + 3 * origins[:, 0].std()])
    plt.savefig('test_atom_cloud.pdf')
    plt.close()

    return 0


def _test_lens(nb_rays=50, f=0.05, m=0.15, right_of_lens=True):
    lens = optics.PerfectLens(f=f, m=m)

    # Create rays parallel to the optical axis
    z_pos = torch.linspace(-lens.f * lens.na / 2 + 1e-5, lens.f * lens.na / 2 - 1e-5, nb_rays)
    origins = batch_vector(torch.zeros(nb_rays) + (1000 if right_of_lens else -1000),
                           torch.zeros(nb_rays),
                           z_pos)

    directions = batch_vector(torch.ones(nb_rays) * (-1 if right_of_lens else 1),
                              torch.zeros(nb_rays),
                              torch.zeros(nb_rays))

    rays = optics.Rays(origins, directions)
    t = lens.get_ray_intersection(rays)
    refracted_rays = lens.intersect(rays, t)

    # check if all rays intersect at the focal length
    o = refracted_rays.origins
    d = refracted_rays.directions
    f_ = f if right_of_lens else -f
    t = (o[:, 0] - f_) / d[:, 0]

    pt = torch.empty((nb_rays, 3))
    pt[:, 0] = o[:, 0] + t * d[:, 0]
    pt[:, 1] = o[:, 1] + t * d[:, 1]
    pt[:, 2] = o[:, 2] + t * d[:, 2]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    for i in range(o.shape[0]):
        ax.plot([o[i, 0] - directions[i, 0] / 10, o[i, 0]],
                [o[i, 1], o[i, 1]],
                [o[i, 2], o[i, 2]])
    for i in range(o.shape[0]):
        ax.plot([o[i, 0], pt[i, 0]],
                [o[i, 1], pt[i, 1]],
                [o[i, 2], pt[i, 2]])
    plt.savefig('test_lens.pdf')
    plt.close()

    pt_mean = pt.mean(dim=0)
    error = 0
    for point in pt:
        error += ((point - pt_mean) ** 2).sum().sqrt()

    if (error / nb_rays) < 5e-3:
        return 0
    else:
        return -1


def _ray_marching_test_sensor(eps=1e-3):
    f = 0.05
    m = 0.15

    sensor = optics.Sensor(position=(-f * (1 + m), 0, 0))
    pts = sensor.sample_points_on_sensor(1000000)

    plt.hist2d(pts[:, 1].numpy(), pts[:, 2].numpy(), bins=40)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig('test_ray_marching_sensor.pdf')
    plt.close()

    pts[:, 1] /= sensor.pixel_size[1]
    pts[:, 2] /= sensor.pixel_size[0]

    if ((pts[:, 1].max() - eps) < sensor.resolution[1] / 2) and \
            ((pts[:, 1].min() + eps) > -sensor.resolution[1] / 2) and \
            ((pts[:, 2].max() - eps) < sensor.resolution[0] / 2) and \
            ((pts[:, 2].min() + eps) > -sensor.resolution[0] / 2):
        return 0
    else:
        return -1


def _ray_marching_test_lens(eps=1e-3):
    f = 0.05
    na = 1 / 1.4

    lens = optics.PerfectLens(f=f, na=na)
    pts = lens.sample_points_on_lens(1000000)

    plt.figure()
    plt.hist2d(pts[:, 1].numpy(), pts[:, 2].numpy(), bins=40)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig('test_ray_marching_lens.pdf')
    plt.close()

    if (pts.max() <= (f * na / 2 + eps)) and (pts.min() >= (-f * na / 2 - eps)):
        return 0
    else:
        return -1


def _test_window():
    obj_x_pos = 0.31
    left_interface_x_position = obj_x_pos + .056
    right_interface_x_position = left_interface_x_position + .05
    window = optics.Window(left_interface_x_position, right_interface_x_position)
    atom_cloud = optics.LightSourceFromDistribution(optics.AtomCloud(position=torch.tensor([obj_x_pos, 0., 0.])))
    nb_atoms = int(1e5)

    rays = atom_cloud.sample_rays(nb_atoms)
    t = window.get_ray_intersection(rays)
    r = rays.get_at(~torch.isnan(t))
    t = t[~torch.isnan(t)]
    rays = window.intersect(r, t)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    n = 100
    window.plot(ax)
    rays.plot(ax, [.1 for _ in range(n)], color='k')
    r.plot(ax, t[:n])
    plt.savefig('test_window.pdf')
    plt.close()

    # The refracted direction should be the same as the incident direction
    if ((rays.directions[:n] - r.directions[:n]).abs().sum() / n) < 1e-5:
        return 0
    else:
        return -1


def _test_cross_product():
    assert (cross_product(torch.tensor([1, 0, 0]), torch.tensor([0, 1, 0])) == torch.tensor([0, 0, 1])).all()
    assert (cross_product(torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])) == torch.tensor([1, 0, 0])).all()
    assert (cross_product(torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])) == torch.tensor([0, 1, 0])).all()
    assert (cross_product(torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0])) == torch.tensor([0, 0, -1])).all()
    assert (cross_product(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])) == torch.tensor([-1, 0, 0])).all()
    assert (cross_product(torch.tensor([1, 0, 0]), torch.tensor([0, 0, 1])) == torch.tensor([0, -1, 0])).all()
    return 0


def _test_look_at_transform():
    up = torch.tensor([0., 0., 1.])
    viewing_direction = torch.tensor([1., 0., 0.])
    pos = torch.tensor([0., 0., 0.])
    camera_to_world, world_to_camera = get_look_at_transform(viewing_direction, pos, up=up)

    assert (camera_to_world == torch.tensor([[0., 0., 1., 0.],
                                             [1., 0., 0., 0.],
                                             [0., 1., 0., 0.],
                                             [0., 0., 0., 1.]])).all()

    up = torch.tensor([0., 0., 1.])
    viewing_direction = torch.tensor([-1., 0., 0.])
    pos = torch.tensor([0., 0., 0.])
    camera_to_world, world_to_camera = get_look_at_transform(viewing_direction, pos, up)

    assert (camera_to_world == (torch.tensor([[0., 0., -1., 0.],
                                              [-1., 0., 0., 0.],
                                              [0., 1., 0., 0.],
                                              [0., 0., 0., 1.]]))).all()

    up = torch.tensor([1., 0., 0.])
    viewing_direction = torch.tensor([0., 0., 1.])
    pos = torch.tensor([0., 0., 0.])
    camera_to_world, world_to_camera = get_look_at_transform(viewing_direction, pos, up)

    assert (camera_to_world == (torch.tensor([[0., 1., 0., 0.],
                                              [-1., 0., 0., 0.],
                                              [0., -0., 1., 0.],
                                              [0., 0., 0., 1.]]))).all()

    up = torch.tensor([-1., 0., 0.])
    viewing_direction = torch.tensor([0., 0., 1.])
    pos = torch.tensor([0., 0., 0.])
    camera_to_world, world_to_camera = get_look_at_transform(viewing_direction, pos, up)

    assert (camera_to_world == (torch.tensor([[0., -1., 0., 0.],
                                              [1., 0., 0., 0.],
                                              [-0., 0., 1., 0.],
                                              [0., 0., 0., 1.]]))).all()
    return 0


def _test_grad_rays():
    def toy_intersect(rays: optics.Rays, normal=torch.tensor([.2, .2, .6])) -> optics.Rays:
        x = (rays.directions[:, 0] * normal[0]).reshape(-1, 1)
        y = (rays.directions[:, 1] * normal[1]).reshape(-1, 1)
        z = (rays.directions[:, 2] * normal[2]).reshape(-1, 1)
        new_directions = torch.cat((x, y, z), dim=1)
        new_origins = rays.origins * 2 + 1
        return optics.Rays(new_origins, new_directions)

    origins = torch.randn(1, 3, requires_grad=True)
    directions = torch.randn(1, 3, requires_grad=True)
    rays1 = optics.Rays(origins, directions)
    rays2 = toy_intersect(rays1)
    assert directions.is_leaf
    assert origins.is_leaf

    toy_loss = (rays2.origins * rays2.directions).sum()
    toy_loss.backward()
    assert origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


def _test_grad_mirror_wrt_incident_rays(mirror_position=torch.ones(3),
                                        ray_origins=torch.randn(2, 3, requires_grad=True)):
    mirror = optics.Mirror(mirror_position[0], mirror_position[1], mirror_position[2], torch.tensor([.2, .2, .6]), .005)

    directions = [(mirror_position[0] - ray_origins[:, 0]).reshape(-1, 1),
                  (mirror_position[1] - ray_origins[:, 1]).reshape(-1, 1),
                  (mirror_position[2] - ray_origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    directions = directions.clone().detach().requires_grad_(True)
    rays = optics.Rays(ray_origins, directions)
    t = mirror.get_ray_intersection(rays)

    outgoing_rays = mirror.intersect(rays, t)

    loss = (outgoing_rays.origins * outgoing_rays.directions).sum()
    loss.backward()

    assert ray_origins.is_leaf
    assert directions.is_leaf
    assert ray_origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


def _test_grad_lens_wrt_incident_rays(lens_position=torch.ones(3), ray_origins=torch.randn(2, 3, requires_grad=True)):
    lens = optics.PerfectLens(position=lens_position)
    directions = [(lens_position[0] - ray_origins[:, 0]).reshape(-1, 1),
                  (lens_position[1] - ray_origins[:, 1]).reshape(-1, 1),
                  (lens_position[2] - ray_origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    directions = directions.clone().detach().requires_grad_(True)
    rays = optics.Rays(ray_origins, directions)
    t = lens.get_ray_intersection(rays)

    outgoing_rays = lens.intersect(rays, t)

    loss = (outgoing_rays.origins * outgoing_rays.directions).sum()
    loss.backward()

    assert directions.is_leaf
    assert ray_origins.is_leaf
    assert ray_origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


def _test_grad_sensor_wrt_incident_rays(sensor_position=torch.ones(3),
                                        ray_origins=torch.randn(2, 3, requires_grad=True)):
    sensor = optics.Sensor(position=sensor_position)
    directions = [(sensor_position[0] - ray_origins[:, 0]).reshape(-1, 1),
                  (sensor_position[1] - ray_origins[:, 1]).reshape(-1, 1),
                  (sensor_position[2] - ray_origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    directions = directions.clone().detach().requires_grad_(True)
    rays = optics.Rays(ray_origins, directions)
    t = sensor.get_ray_intersection(rays)

    hit_position, _ = sensor.intersect(rays, t)

    target_position = torch.randn(hit_position.shape)

    l1_loss = ((hit_position - target_position)**2).mean()
    l1_loss.backward()

    assert directions.is_leaf
    assert ray_origins.is_leaf
    assert ray_origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


def _test_grad_mirror_wrt_self_parameters():
    mirror_normal = torch.tensor([.2, .2, .6], requires_grad=True)
    mirror_position = torch.tensor([1., 1., 1.], requires_grad=True)
    mirror = optics.Mirror(mirror_position[0], mirror_position[1], mirror_position[2], mirror_normal, .005)

    origins = torch.randn(2, 3)
    directions = [(mirror_position[0].item() - origins[:, 0]).reshape(-1, 1),
                  (mirror_position[1].item() - origins[:, 1]).reshape(-1, 1),
                  (mirror_position[2].item() - origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    rays = optics.Rays(origins, directions)
    t = mirror.get_ray_intersection(rays)
    outgoing_rays = mirror.intersect(rays, t)

    # Toy loss
    loss = (outgoing_rays.origins * outgoing_rays.directions).sum()
    loss.backward()

    assert mirror_normal.is_leaf
    assert mirror_position.is_leaf
    assert mirror_position.grad.abs().sum() > 0
    assert mirror_normal.grad.abs().sum() > 0
    return 0


def _test_grad_lens_wrt_self_parameters():
    return 0


def _test_grad_sensor_wrt_self_parameters():
    return 0


"""
            Tests for pytest
"""


def test_ray():
    assert _test_ray(1) == 0
    assert _test_ray(1000) == 0


def test_rejection_sampling():
    assert _test_rejection_sampling() == 0


def test_atom_cloud():
    assert _test_atom_cloud() == 0


def test_lens():
    assert _test_lens(f=0.05, m=0.15) == 0
    assert _test_lens(f=0.04, m=0.15) == 0
    assert _test_lens(f=0.01, m=0.15) == 0
    assert _test_lens(f=0.05, m=0.1) == 0
    assert _test_lens(f=0.04, m=0.1) == 0
    assert _test_lens(f=0.01, m=0.1) == 0

    assert _test_lens(f=0.05, m=0.15, right_of_lens=False) == 0
    assert _test_lens(f=0.01, m=0.15, right_of_lens=False) == 0
    assert _test_lens(f=0.05, m=0.1, right_of_lens=False) == 0


def test_ray_marching():
    assert _ray_marching_test_sensor() == 0
    assert _ray_marching_test_lens() == 0


def test_window():
    assert _test_window() == 0


def test_gradients_wrt_incident_rays():
    """
    For each optical component, check that the gradients with respect the incident rays propagate correctly
    """
    assert _test_grad_rays() == 0
    assert _test_grad_mirror_wrt_incident_rays() == 0
    assert _test_grad_lens_wrt_incident_rays() == 0
    assert _test_grad_sensor_wrt_incident_rays() == 0


def test_gradients_wrt_self_parameters():
    """
    For each optical component, check that the gradients with respect its parameters propagate correctly
    """
    assert _test_grad_mirror_wrt_self_parameters() == 0
    assert _test_grad_lens_wrt_self_parameters() == 0
    assert _test_grad_sensor_wrt_self_parameters() == 0


def test_vectors():
    assert _test_cross_product() == 0


def test_transforms():
    assert _test_look_at_transform() == 0
